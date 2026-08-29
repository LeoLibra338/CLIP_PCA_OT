"""
OpenCLIP/TULIP (ViT-L/14) 在 COCO train2017 上 fine-tune
使用 open_clip 库加载 ViT-L/14 backbone，标准 InfoNCE loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
import sys, os, argparse, numpy as np

sys.path.insert(0, '/home/gyf/CLIP-wgd/open_clip/src')
from open_clip import create_model_and_transforms, get_tokenizer

from torch.utils.data.distributed import DistributedSampler
from scheduler import cosine_lr
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from sharegpt4v import share4v_train_dataset


def all_gather_with_grad(tensor):
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


class AverageMeter:
    def __init__(self, length=0):
        self.length = length; self.history = []; self.sum = 0; self.count = 0
    def update(self, val, num=1):
        if self.length > 0:
            self.history.append(val)
            if len(self.history) > self.length: self.history.pop(0)
        else: self.sum += val * num; self.count += num
    @property
    def avg(self):
        if self.length > 0: return np.mean(self.history) if self.history else 0
        return self.sum / self.count if self.count > 0 else 0


class OpenCLIPTrainer:
    def __init__(self, rank, local_rank, args):
        self.rank = rank; self.local_rank = local_rank

        # 加载 open_clip ViT-L/14
        self.model, _, self.preprocess = create_model_and_transforms(
            'ViT-L-14', pretrained=args.base_model
        )
        self.model = self.model.float()
        self.model.train()
        self.model = self.model.cuda()

        self.batch_size = args.batch_size
        self.num_epoch = args.epochs
        self.accum_steps = args.accum_steps

        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[local_rank], find_unused_parameters=True
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()

        # open_clip tokenizer
        self.tokenizer = get_tokenizer('ViT-L-14')

        jobname = args.jobname
        self.save_dir = os.path.join(args.save_root, jobname)
        if rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            print(vars(args))

        self.args = args

    def train_epoch(self, dataloader, epoch, train_losses):
        print_freq = 20
        accum_steps = self.accum_steps
        num_batches = len(dataloader)
        self.optimizer.zero_grad()

        for i, (images, texts, _) in enumerate(tqdm(dataloader, disable=self.rank != 0)):
            step = num_batches * epoch + i
            self.scheduler(step)

            images = images.cuda()
            # open_clip tokenizer 返回 padded token tensor
            text_tokens = self.tokenizer(texts, context_length=77).cuda()

            with torch.cuda.amp.autocast():
                image_features = self.model.module.encode_image(images)
                text_features = self.model.module.encode_text(text_tokens)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                img_all = all_gather_with_grad(image_features)
                txt_all = all_gather_with_grad(text_features)

                sim_i2t = image_features @ txt_all.T
                sim_t2i = img_all @ text_features.T
                sim_t2i = sim_t2i.T

                logit_scale = self.model.module.logit_scale.exp()
                sim_i2t = logit_scale * sim_i2t
                sim_t2i = logit_scale * sim_t2i

                bs = images.size(0)
                targets = torch.linspace(self.rank * bs, self.rank * bs + bs - 1,
                                         bs, dtype=torch.long).cuda()
                loss = (F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) +
                        F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)) / 2
                loss = loss / accum_steps

            train_losses.update((loss * accum_steps).detach().item())
            self.scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == num_batches:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            if self.rank == 0 and i % print_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f'EP:{epoch}/{self.num_epoch-1} step:{step} loss:{train_losses.avg:.4f} lr:{lr:.2e}')

    def train(self):
        trainset = share4v_train_dataset()
        sampler = DistributedSampler(dataset=trainset, shuffle=True)
        loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                             sampler=sampler, num_workers=4, pin_memory=True)
        train_losses = AverageMeter(20)
        self.scheduler = cosine_lr(self.optimizer, base_lr=self.args.lr,
                                   warmup_length=self.args.warmup_length,
                                   steps=self.num_epoch * len(loader))
        for epoch in range(self.num_epoch):
            self.train_epoch(loader, epoch, train_losses)
            if self.rank == 0:
                torch.save(self.model.module.state_dict(),
                           os.path.join(self.save_dir, f'ep={epoch}_openclip.pt'))


def setup_distributed(backend="nccl"):
    num_gpus = torch.cuda.device_count()
    rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    return rank, rank % num_gpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--log_scale', type=float, default=4.6052)
    parser.add_argument('--jobname', type=str, default='openclip_coco')
    parser.add_argument('--save_root', type=str, default='./output/ckpts')
    parser.add_argument('--warmup_length', type=int, default=200)
    parser.add_argument('--base_model', type=str, default='../weights/ViT-L-14.pt')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--accum_steps', type=int, default=4)
    args = parser.parse_args()

    rank, local_rank = setup_distributed()
    trainer = OpenCLIPTrainer(rank, local_rank, args)
    trainer.train()
