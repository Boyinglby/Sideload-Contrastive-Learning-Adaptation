import os
import random
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------------------
# Config
# ---------------------------
FRAME_DIR = "dataset_unanotated/frames"   
BATCH_SIZE = 8
NUM_WORKERS = 4
IMG_SIZE = (1920, 1920)          # (W, H) fixed so encoder feature grid is consistent
LR_HEAD = 1e-3
LR_ENC  = 1e-4
NUM_EPOCHS = 30
TEMP = 0.1
MIXED_PREC = torch.cuda.is_available()

# Patch config (on encoder feature map)
PATCH_H, PATCH_W = 2, 2
STRIDE_H, STRIDE_W = 1, 1      # overlap to better capture tiny objects

# ---------------------------
# Dataset & transforms
# ---------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class PairDataset(Dataset):
    """
    Returns (img1_tensor, img2_tensor, label)
      label: 0 = similar (same frame, two photometric augs)
             1 = dissimilar (current frame + random *different* frame)
    """
    def __init__(self, frame_dir: str, transform: transforms.Compose, exclude_prefix: str = "frame2022-12-04 Bjenberg"):
        self.paths = self._load_frame_paths(frame_dir, exclude_prefix)

        if len(self.paths) == 0:
            raise FileNotFoundError(f"No .png files found under {frame_dir}")
        self.transform = transform

    def _load_frame_paths(self, frame_dir: str, exclude_prefix: str) -> List[str]:
        return sorted(
            os.path.join(frame_dir, f)
            for f in os.listdir(frame_dir)
            if f.lower().endswith(".png") and not f.startswith(exclude_prefix)
        )

    def __len__(self):
        return len(self.paths)

    def _load_pil_rgb(self, path: str) -> Image.Image:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize(IMG_SIZE, Image.BILINEAR)  # Pillow expects (W, H)
            return img

    def __getitem__(self, idx: int):
        # 50/50 similar vs dissimilar
        make_similar = (random.random() < 0.5)

        path1 = self.paths[idx]
        img1 = self._load_pil_rgb(path1)

        if make_similar:
            # same frame, different photometric aug
            img2 = img1.copy()
            label = 0
        else:
            # pick a different frame path
            path2 = path1
            while path2 == path1:
                path2 = random.choice(self.paths)
            img2 = self._load_pil_rgb(path2)
            label = 1

        # Photometric augs only (keep spatial alignment)
        x1 = self.transform(img1)
        x2 = self.transform(img2)
        return x1, x2, label

photometric_transform = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------------------
# Patch extraction (with overlap)
# ---------------------------
def feature_map_patch(x: torch.Tensor,
                      window_h: int = 2, window_w: int = 2,
                      stride_h: Optional[int] = None, stride_w: Optional[int] = None) -> Tuple[torch.Tensor, Tuple[int,int]]:
    """
    x: [B,C,H,W]
    Returns:
        patches_flat: [B, P, C*window_h*window_w]
        grid: (Hp, Wp)
    """
    B, C, H, W = x.shape
    ph, pw = window_h, window_w
    sh = ph if stride_h is None else stride_h
    sw = pw if stride_w is None else stride_w

    assert H >= ph and W >= pw, "window larger than feature map"
    assert (H - ph) % sh == 0 and (W - pw) % sw == 0, \
        f"Feature map {H}x{W} not compatible with window {ph}x{pw} and stride {sh}x{sw}"

    patches = x.unfold(2, ph, sh).unfold(3, pw, sw)        # [B, C, Hp, Wp, ph, pw]
    B_, C_, Hp, Wp, ph_, pw_ = patches.shape
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()   # [B, Hp, Wp, C, ph, pw]
    patches_flat = patches.view(B_, Hp * Wp, C_ * ph_ * pw_)   # [B, P, Din]
    return patches_flat, (Hp, Wp)


def freeze_bn_and_params(modules: List[nn.Module]):
    for m in modules:
        for p in m.parameters():
            p.requires_grad = False
        for sub in m.modules():
            if isinstance(sub, nn.BatchNorm2d):
                sub.eval()
                sub.requires_grad_(False)

class CL_YOLO_Patch(nn.Module):
    def __init__(self, layers: List[nn.Module],
                 feature_dim: int = 64,
                 patch_h: int = 2, patch_w: int = 2,
                 stride_h: int = 1, stride_w: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(*layers)
        self.g = nn.Sequential(
            nn.LazyLinear(512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )
        self.patch_h, self.patch_w = patch_h, patch_w
        self.stride_h, self.stride_w = stride_h, stride_w

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int]]:
        feats = self.encoder(x)  # [B,C,Hf,Wf]
        patches_flat, (Hp, Wp) = feature_map_patch(
            feats, self.patch_h, self.patch_w, self.stride_h, self.stride_w
        )  # [B,P,Din]
        B, P, Din = patches_flat.shape
        z = self.g(patches_flat.view(B * P, Din))   # [B*P, D]
        z = F.normalize(z, dim=1)                   # unit norm
        z = z.view(B, P, -1)                        # [B,P,D]
        return z, (Hp, Wp)


def _patch_pos_mask_indices(pos_batch_idx: torch.Tensor, pool_batch_idx: torch.Tensor, P: int) -> torch.Tensor:
    """
    Build a block-diagonal boolean mask (size [Npos*P, Npool*P]) with identity P on matching image pairs.
    """
    device = pos_batch_idx.device
    Npos = pos_batch_idx.numel()
    Npool = pool_batch_idx.numel()

    pos_in_pool = (pos_batch_idx.view(-1,1) == pool_batch_idx.view(1,-1))  # [Npos,Npool]
    pool_positions = pos_in_pool.float().argmax(dim=1)  # [Npos], assumes unique match
    eyeP = torch.eye(P, dtype=torch.bool, device=device)

    mask = torch.zeros((Npos * P, Npool * P), dtype=torch.bool, device=device)
    for i in range(Npos):
        r0 = i * P
        c0 = pool_positions[i].item() * P
        mask[r0:r0+P, c0:c0+P] = eyeP
    return mask

def patch_info_nce(z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor, temp: float = 0.1) -> Optional[torch.Tensor]:
    """
    z1, z2: [B,P,D]
    labels: [B], 0 = similar pair (same image), 1 = dissimilar
    Use only similar pairs as anchors; all patches from entire batch act as negatives.

    Returns:
        loss tensor if there are positives, otherwise None (skip batch).
    """
    B, P, D = z1.shape
    device = z1.device
    pos_mask_batch = (labels == 0)

    if pos_mask_batch.sum() == 0:
        return None  # skip this batch

    z1_all = z1.view(B * P, D)
    z2_all = z2.view(B * P, D)

    pos_idx = torch.nonzero(pos_mask_batch, as_tuple=False).squeeze(1)  # [Npos]
    pos_mask = _patch_pos_mask_indices(pos_idx, torch.arange(B, device=device), P)  # [Npos*P, B*P]

    z1_pos = z1[pos_idx].contiguous().view(-1, D)  # [Npos*P, D]
    sim = (z1_pos @ z2_all.T) / temp               # [Npos*P, B*P]
    sim = sim - sim.max(dim=1, keepdim=True)[0]    # stabilize
    exp_sim = torch.exp(sim)

    pos_sum = (exp_sim * pos_mask).sum(dim=1) + 1e-12
    all_sum = exp_sim.sum(dim=1) + 1e-12
    loss_a2b = -torch.log(pos_sum / all_sum).mean()

    # Symmetrize (anchor z2 -> pool z1)
    z2_pos = z2[pos_idx].contiguous().view(-1, D)
    sim2 = (z2_pos @ z1_all.T) / temp
    sim2 = sim2 - sim2.max(dim=1, keepdim=True)[0]
    exp_sim2 = torch.exp(sim2)
    pos_sum2 = (exp_sim2 * pos_mask).sum(dim=1) + 1e-12
    all_sum2 = exp_sim2.sum(dim=1) + 1e-12
    loss_b2a = -torch.log(pos_sum2 / all_sum2).mean()

    return 0.5 * (loss_a2b + loss_b2a)

def build_model() -> CL_YOLO_Patch:
    yolo = YOLO("yolo11n.pt")
    all_layers = list(yolo.model.model.children())

    # Choose a high-resolution slice (early layers) to preserve small-object detail
    # freeze_layers = all_layers[:7]
    # freeze_bn_and_params(freeze_layers)
    # unfreeze_layers = all_layers[7:9]

    unfreeze_layers = all_layers[:9]

    
    for l in unfreeze_layers:
        for p in l.parameters():
            p.requires_grad = True

    # layers = freeze_layers + unfreeze_layers
    layers = unfreeze_layers
    model = CL_YOLO_Patch(
        layers,
        feature_dim=64,
        patch_h=PATCH_H, patch_w=PATCH_W,
        stride_h=STRIDE_H, stride_w=STRIDE_W
    )
    return model

def make_optimizer(model: CL_YOLO_Patch) -> optim.Optimizer:
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("g.")]
    enc_params  = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("g.")]
    optimizer = optim.Adam([
        {"params": enc_params,  "lr": LR_ENC},
        {"params": head_params, "lr": LR_HEAD},
    ])
    return optimizer



# ---------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, optimizer, epoch: int, epochs: int):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PREC)
    running = 0.0
    num_updates = 0

    bar = tqdm(loader)
    for img1, img2, label in bar:
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True).long().view(-1)

        optimizer.zero_grad(set_to_none=True)

        if MIXED_PREC:
            with torch.cuda.amp.autocast():  # compatible with older PyTorch
                z1, grid1 = model(img1)  # [B,P,D]
                z2, grid2 = model(img2)  # [B,P,D]
                assert grid1 == grid2, f"Feature grids must match: {grid1} vs {grid2}"
                loss = patch_info_nce(z1, z2, label, temp=TEMP)
        else:
            z1, grid1 = model(img1)
            z2, grid2 = model(img2)
            assert grid1 == grid2, f"Feature grids must match: {grid1} vs {grid2}"
            loss = patch_info_nce(z1, z2, label, temp=TEMP)

        # Skip steps when there is nothing to learn this batch
        if loss is None:
            bar.set_description(f"Epoch [{epoch+1}/{epochs}] Loss: (skip)")
            continue

        if MIXED_PREC:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # REQUIRED before clipping/step for AMP bookkeeping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        num_updates += 1
        running += float(loss.detach().cpu())
        bar.set_description(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

    return running / max(1, num_updates)

def main():
    parser = argparse.ArgumentParser(description="Patch-level CL pretraining for YOLO features")
    parser.add_argument("--frame-dir", default=FRAME_DIR, help="Directory containing frame PNG files")
    parser.add_argument("--exclude-prefix", default="frame2022-12-04 Bjenberg",
                        help="Exclude frame files that start with this prefix")
    args = parser.parse_args()

    best_epoch = 1
    dataset = PairDataset(args.frame_dir, transform=photometric_transform, exclude_prefix=args.exclude_prefix)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # 1) Build and move model
    model = build_model().to(device)

    # 2) Run a dummy forward ONCE to materialize LazyLinear params
    with torch.no_grad():
        x_dummy = torch.zeros(1, 3, IMG_SIZE[1], IMG_SIZE[0], device=device)  # [B,C,H,W]
        _ = model(x_dummy)

    # 3) Now create the optimizer (after lazy layers are initialized)
    optimizer = make_optimizer(model)

    best_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, loader, optimizer, epoch, NUM_EPOCHS)

        # save every epoch (state_dict includes encoder slice + projector)
        torch.save(model.encoder, "patchCL_yolo_projector.pt")

        if train_loss < best_loss:
            best_epoch = epoch + 1
            best_loss = train_loss
            torch.save(model.encoder, "patchCL_yolo_backbone_best.pt")

    print("Done. Best train loss:", best_epoch, best_loss)

if __name__ == "__main__":
    main()
