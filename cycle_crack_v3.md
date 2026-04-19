# System Instruction: Cycle-Crack Implementation Agent (v2 — Paper-Accurate)

## Role
You are an expert PyTorch engineer. Implement the **Cycle-Crack** framework **exactly** as described in the paper:
> *"Cycle-Crack: Weakly Supervised Pavement Crack Segmentation via Dual-Cycle Generative Adversarial Learning"*

Every implementation decision must trace back to the paper. Where the paper is silent, use the most conservative standard choice and mark it with `# [not in paper]`.

---

## Project Structure

```
cycle_crack/
├── config.py       # All hyperparameters
├── model.py        # Generator (UNet-256) + Discriminator (PatchGAN)
├── loss.py         # All 5 loss functions
├── dataset.py      # Sliding-window patch dataset
├── train.py        # Dual-cycle training loop
├── infer.py        # Inference + crack map pipeline
└── evaluate.py     # Metrics: P, R, F1, IoU, mBF1
```

---

## 1. `config.py` — Hyperparameters (strict paper values)

```python
# ── Architecture ──
IMG_SIZE      = 256
IN_CHANNELS   = 3
OUT_CHANNELS  = 3
NGF           = 64          # generator base filters
NDF           = 64          # discriminator base filters
N_ENC_LAYERS  = 8           # encoder depth (UNet-256)

# ── Training ──
BATCH_SIZE    = 16
EPOCHS        = 100
LR_G          = 1e-4        # generator learning rate  [paper §IV-B]
LR_D          = 4e-4        # discriminator learning rate [paper §IV-B]
BETA1         = 0.5
BETA2         = 0.999
SCHEDULER     = "cosine"    # CosineAnnealingLR for both G and D

# ── Loss weights ──  [paper Eq.2 and §IV-B]
LAMBDA_IDT    = 0.5
LAMBDA_CYCLE  = 10.0
LAMBDA_ADV    = 1.0
LAMBDA_TEXT   = 2.0         # ← 2.0, NOT 5.0
LAMBDA_REGION = 1.0         # ← 1.0, NOT 10.0

# ── Identity sub-weights (paper Eq.3) ──
LAMBDA_GE     = 1.0         # weight for GE identity term
LAMBDA_GA     = 1.0         # weight for GA identity term

# ── Cycle sub-weights (paper Eq.5) ──
LAMBDA_C      = 1.0         # crack-elimination cycle branch
LAMBDA_N      = 1.0         # crack-addition cycle branch

# ── Dataset ──
CRACK_PIXEL_THRESHOLD = 100   # patch must have >100 crack pixels [paper §IV-A]
STRIDE_CRACK500       = 256
STRIDE_DEEPCRACK      = 128
STRIDE_CFD            = 128

# ── Inference ──
BILATERAL_D           = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
```

---

## 2. `model.py` — Architecture

### 2.1 Generator: `UNet256`

**CRITICAL**: The paper explicitly states **BatchNorm2d**, NOT InstanceNorm2d.

```
Encoder (8 downsampling blocks):
  Each block: Conv2d(4×4, stride=2, padding=1) → BatchNorm2d → LeakyReLU(0.2)
  Channel progression: [64, 128, 256, 512, 512, 512, 512, 512]
  First encoder block: NO BatchNorm (paper standard for UNet-256)

Decoder (8 upsampling blocks):
  Each block: ConvTranspose2d(4×4, stride=2, padding=1) → BatchNorm2d → ReLU
  Channel progression (after concat with skip): mirrors encoder
  Dropout(p=0.5) applied to decoder layers 1, 2, 3 (innermost three)
  Skip connections: concat encoder[i] with decoder[8-i] along channel dim

Final layer: Conv2d(1×1) → Tanh  →  output ∈ [-1, 1]
```

Instantiate **two** generators:
- `G_E`: crack elimination  (C → N)
- `G_A`: crack addition     (N → C)

Both share the same `UNet256` class with identical hyperparameters.

```python
class UNet256(nn.Module):
    """
    UNet-256 generator as described in paper §III-A.
    Uses BatchNorm2d (not InstanceNorm) and skip-connections via concatenation.
    Dropout(0.5) in the 3 innermost decoder layers.
    """
```

### 2.2 Discriminator: `PatchGAN70`

The paper specifies a **70×70 PatchGAN** with exactly **5 convolutional layers**:

```
Layer 1: Conv2d(in, 64,  4, stride=2, pad=1) → LeakyReLU(0.2)   [no BN]
Layer 2: Conv2d(64, 128, 4, stride=2, pad=1) → BN → LeakyReLU(0.2)
Layer 3: Conv2d(128,256, 4, stride=2, pad=1) → BN → LeakyReLU(0.2)
Layer 4: Conv2d(256,512, 4, stride=1, pad=1) → BN → LeakyReLU(0.2)
Layer 5: Conv2d(512,  1, 4, stride=1, pad=1)                      [no BN, no act]
```

**NOTE**: The paper does NOT mention Spectral Normalization. Do NOT add it.

`forward()` must return **both** the final output AND the intermediate feature map from **layer 4** (used by `L_region`):

```python
def forward(self, x):
    feat = None
    for i, layer in enumerate(self.layers):
        x = layer(x)
        if i == 3:          # layer 4 (0-indexed)
            feat = x        # shape: (B, 512, H', W')
    return x, feat          # (patch_pred, intermediate_features)
```

Instantiate **two** discriminators:
- `D_N`: judges "is this a real normal (crack-free) image?"
- `D_C`: judges "is this a real cracked image?"

---

## 3. `loss.py` — All 5 Loss Functions

### 3.1 Identity Loss `L_idt`  [paper Eq.3]

```
L_idt = lambda_GE * ||G_E(I_C) - I_N||_1
      + lambda_GA * ||G_A(I_N) - I_C||_1
```

### 3.2 Adversarial Loss `L_adv`  [paper Eq.4] — LSGAN (MSE)

```
# Discriminator update
L_D = E[(D_N(I_N) - 1)²] + E[D_N(G_E(I_C).detach())²]
    + E[(D_C(I_C) - 1)²] + E[D_C(G_A(I_N).detach())²]

# Generator update (fool discriminators)
L_G_adv = E[(D_N(G_E(I_C)) - 1)²]
         + E[(D_C(G_A(I_N)) - 1)²]
```

**CRITICAL**: Use `.detach()` on generator outputs when computing `L_D` to prevent gradients flowing back through G.

### 3.3 Cycle Consistency Loss `L_cycle`  [paper Eq.5]

```
I'_C  = G_E(I_C)         # crack → crack-free
I''_C = G_A(I'_C)        # crack-free → crack (should ≈ I_C)

I'_N  = G_A(I_N)         # crack-free → crack
I''_N = G_E(I'_N)        # crack → crack-free (should ≈ I_N)

L_cycle = lambda_C * ||I''_C - I_C||_1
        + lambda_N * ||I''_N - I_N||_1
```

### 3.4 Texture Consistency Loss `L_text`  [paper §III-C]

**Applied to BOTH cycle directions** (C→N→C and N→C→N).

Combines 3 sub-terms:

#### a) MS-SSIM Loss  [paper Eq.6–7]
```
L_ms_ssim = (1/S) * Σ_s (1 - SSIM(I_real_s, I_rec_s))

SSIM(x,y) = [l(x,y)]^α * [c(x,y)]^β * [s(x,y)]^γ
where α = β = γ = 1
```
Use S=3 scales (downsample by 2× each time). Implement SSIM with 11×11 Gaussian window.

#### b) MSGMS Loss  [paper Eq.8]
```
L_msgms = (1/S) * Σ_s ||GM(I_real_s) - GM(I_rec_s)||_1

GM(x) = sqrt((∂x/∂x)² + (∂x/∂y)²)   ← Sobel gradient magnitude
```
Implement Sobel using `F.conv2d` with fixed Sobel kernels (differentiable).
Apply across S=3 scales.

#### c) Style Loss  [paper Eq.9–10]
```
L_style = Σ_l ||Gram(VGG_l(I_real)) - Gram(VGG_l(I_rec))||²_F

Gram(x) = x @ x.T / (C × H × W)
```

**CRITICAL — VGG layer specification** [paper §III-C]: The paper states `"with l = 3 used in this work"`.
This means **a single VGG layer at index 3** — NOT 4 layers, NOT a range [:4].

Implement exactly as:
```python
import torchvision.models as models

vgg = models.vgg16(pretrained=True).features
# Extract features up to and including layer index 3
# VGG-16 features[:4] = [Conv2d, ReLU, Conv2d, ReLU]  ← index 3 is the 2nd ReLU (relu1_2)
# BUT paper means only the OUTPUT at index 3, not a cumulative sum of 4 layers.

# Correct implementation — ONE feature map only:
vgg_layer3 = nn.Sequential(*list(vgg.children())[:4])  # output = relu1_2 features
for p in vgg_layer3.parameters():
    p.requires_grad = False   # freeze

def style_loss(I_real, I_rec):
    feat_real = vgg_layer3(I_real)           # single feature map
    feat_rec  = vgg_layer3(I_rec)
    B, C, H, W = feat_real.shape
    gram_real = feat_real.view(B, C, -1)
    gram_real = torch.bmm(gram_real, gram_real.transpose(1, 2)) / (C * H * W)
    gram_rec  = feat_rec.view(B, C, -1)
    gram_rec  = torch.bmm(gram_rec, gram_rec.transpose(1, 2)) / (C * H * W)
    return F.mse_loss(gram_real, gram_rec)
```

**Stability note** [not in paper — engineering fix]: When using AMP (FP16), the Gram matrix MSE
can overflow. Apply `style_loss * 1e-3` scaling factor OR compute VGG forward pass in FP32
via `with torch.autocast(enabled=False)`. Mark whichever you choose with `# [not in paper]`.

**Do NOT** use VGG-19 or sum across multiple layers — that is NOT in the paper.

#### Combined:
```python
L_text = L_ms_ssim + L_msgms + L_style
# Apply to cycle C→N→C: pair (I_C, I''_C)
# Apply to cycle N→C→N: pair (I_N, I''_N)
```

### 3.5 Region Consistency Loss `L_region`  [paper Eq.11–12]

**CRITICAL**: Applied **only** to the G_E (crack elimination) branch. NOT applied to G_A.

```
# Attention mask — paper Eq.11:
# M_attn = σ( ||F(I_real) - F(I_fake)|| )
#
# CRITICAL identity of I_real and I_fake:
#   I_real = I_N  (a REAL normal/crack-free image fed to D_N)
#   I_fake = G_E(I_C)  (the generated crack-free image from G_E)
#
# Both are passed through D_N (the normal-domain discriminator).
# The feature difference highlights regions where G_E(I_C) deviates
# from the true normal distribution — i.e., crack regions.
#
# Do NOT use I_C as I_real — I_C is the cracked image, not a normal image.

_, feat_real = D_N(I_N)            # features from REAL normal image
_, feat_fake = D_N(G_E(I_C))       # features from GENERATED normal image

# Compute per-spatial-location L1 norm across channel dim, then sigmoid
feat_diff = torch.abs(feat_real - feat_fake)          # (B, C, H', W')
feat_norm = feat_diff.mean(dim=1, keepdim=True)        # (B, 1, H', W')
M_attn    = torch.sigmoid(feat_norm)                   # (B, 1, H', W')
M_attn    = F.interpolate(M_attn, size=I_C.shape[-2:], mode='bilinear', align_corners=False)

# Region loss — paper Eq.12:
# L_region = ||∇G_E(I_C) - ∇I_C||_1  +  ||(G_E(I_C) - I_C) ⊙ (1 - M_attn)||_1
I_fake_C  = G_E(I_C)
grad_fake  = sobel(I_fake_C)     # differentiable Sobel via F.conv2d
grad_real  = sobel(I_C)

L_region = (  F.l1_loss(grad_fake, grad_real)
            + F.l1_loss((I_fake_C - I_C) * (1.0 - M_attn),
                        torch.zeros_like(I_C))  )
```

Where `sobel()` uses fixed Sobel kernels applied via `F.conv2d` (fully differentiable).
Where `∇` is the **Sobel** gradient operator — do NOT use finite difference or torch.gradient.

### 3.6 Total Loss  [paper Eq.2]

```
L_total = lambda_idt    * L_idt
        + lambda_cycle  * L_cycle
        + lambda_adv    * L_adv
        + lambda_text   * L_text       # λ=2.0
        + lambda_region * L_region     # λ=1.0
```

---

## 4. `dataset.py` — Data Pipeline

### Sliding Window Patch Extraction
```
window_size = 256 × 256
stride:
  Crack500  → 256 px
  DeepCrack → 128 px
  CFD       → 128 px
```

### Patch Assignment  [paper §IV-A]
- **Cracked domain**: patch has **> 100 crack pixels** in GT mask
- **Normal domain**: patch has **0 crack pixels** in GT mask
- Patches with 1–100 crack pixels are **discarded** (ambiguous)

### Dataset Split  [paper §IV-A]
- Fixed random seed for reproducibility
- No overlap between train and test sets

### `CrackDataset` class:
```python
class CrackDataset(Dataset):
    """
    Returns unpaired (I_C, I_N) for each training iteration.
    I_N is sampled randomly from the normal pool each epoch.
    Augmentation (flip, rotation, scale+crop) applied to training set only.
    Normalise to [-1, 1].
    """
    def __getitem__(self, idx):
        # Return dict with keys 'cracked' and 'normal'
        return {'cracked': I_C_tensor, 'normal': I_N_tensor}
```

Augmentation (training only): random horizontal flip, vertical flip, rotation (±10°), random scale+crop.

---

## 5. `train.py` — Training Loop

### Optimizer Setup  [paper §IV-B]
```python
optimizer_G = Adam(list(G_E.parameters()) + list(G_A.parameters()),
                   lr=LR_G, betas=(BETA1, BETA2))
optimizer_D = Adam(list(D_N.parameters()) + list(D_C.parameters()),
                   lr=LR_D, betas=(BETA1, BETA2))

scheduler_G = CosineAnnealingLR(optimizer_G, T_max=EPOCHS)
scheduler_D = CosineAnnealingLR(optimizer_D, T_max=EPOCHS)
```

### Training Loop Per Iteration
```
─── Forward (both cycles) ───────────────────────────────
I'_C  = G_E(I_C)                 # eliminate cracks
I''_C = G_A(I'_C)                # reconstruct cracked (cycle 1)

I'_N  = G_A(I_N)                 # add cracks
I''_N = G_E(I'_N)                # reconstruct normal (cycle 2)

I_C_idt = G_E(I_N)               # identity: normal through G_E
I_N_idt = G_A(I_C)               # identity: cracked through G_A

─── Step 1: Update Discriminators ───────────────────────
pred_real_N, _      = D_N(I_N)
pred_fake_N, _      = D_N(I'_C.detach())    # MUST detach
pred_real_C, _      = D_C(I_C)
pred_fake_C, _      = D_C(I'_N.detach())    # MUST detach

loss_D = LSGAN(pred_real_N, real=True)  + LSGAN(pred_fake_N, real=False)
       + LSGAN(pred_real_C, real=True)  + LSGAN(pred_fake_C, real=False)

optimizer_D.zero_grad(); loss_D.backward(); optimizer_D.step()

─── Step 2: Update Generators ───────────────────────────
# Adversarial (fool discriminators — no detach here)
pred_fake_N_G, _      = D_N(I'_C)
pred_fake_C_G, _      = D_C(I'_N)
loss_adv = LSGAN(pred_fake_N_G, real=True) + LSGAN(pred_fake_C_G, real=True)

# Cycle consistency
loss_cycle = lambda_C * L1(I''_C, I_C) + lambda_N * L1(I''_N, I_N)

# Identity
loss_idt = lambda_GE * L1(I_C_idt, I_N) + lambda_GA * L1(I_N_idt, I_C)

# Texture (both cycles)
loss_text = TextureLoss(I_C, I''_C) + TextureLoss(I_N, I''_N)

# Region (G_E only — get D_N features for attention mask)
_, feat_real = D_N(I_N)
_, feat_fake = D_N(I'_C)
loss_region = RegionLoss(I_C, I'_C, feat_real, feat_fake)

loss_G = (LAMBDA_ADV    * loss_adv
        + LAMBDA_CYCLE  * loss_cycle
        + LAMBDA_IDT    * loss_idt
        + LAMBDA_TEXT   * loss_text
        + LAMBDA_REGION * loss_region)

optimizer_G.zero_grad(); loss_G.backward(); optimizer_G.step()

─── Schedulers ──────────────────────────────────────────
scheduler_G.step(); scheduler_D.step()
```

### Checkpoints & Logging
- Save every 10 epochs: `checkpoints/epoch_{n}.pth` (G_E, G_A, D_N, D_C state dicts + optimizers)
- Log per epoch: loss_D, loss_G, loss_adv, loss_cycle, loss_idt, loss_text, loss_region

---

## 6. `infer.py` — Inference Pipeline  [paper §III-B, Eq.1]

```
For a full-resolution input image I_C:

1. Partition into overlapping 256×256 patches
   (use same stride as training for the dataset, or stride=128 for overlap)

2. For each patch p:
   a. Normalize to [-1, 1]
   b. p' = G_E(p)                         # crack-free reconstruction
   c. E = |p - p'|                        # error map (3-channel)
   d. E_gray = E.mean(dim=0)              # collapse to 1 channel

3. Stitch patches back to full resolution
   (average overlapping regions)

4. Rescale E_gray to [0, 255] uint8

5. Apply bilateral filter:
   cv2.bilateralFilter(E_gray_uint8, d=9, sigmaColor=75, sigmaSpace=75)

6. Apply OTSU threshold:
   _, M = cv2.threshold(E_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

7. Return binary mask M  (255=crack, 0=background)
```

**NOTE**: The paper does NOT include morphological closing in the inference pipeline. Do NOT add it unless marked with `# [not in paper]`.

---

## 7. `evaluate.py` — Evaluation Metrics  [paper §IV-B]

Implement the following metrics:

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × P × R / (P + R)
IoU       = TP / (TP + FP + FN)

BF1(d) = 2 × Pb(d) × Rb(d) / (Pb(d) + Rb(d))
  where Pb(d) = boundary precision within tolerance d pixels
        Rb(d) = boundary recall within tolerance d pixels

mBF1 = mean of BF1 at d ∈ {1, 3, 5}  [paper §IV-B, Eq.13]
```

Extract boundaries using morphological gradient.
mBF1 is the **primary** boundary metric in the paper.

---

## 8. Implementation Rules

1. **BatchNorm2d** in Generator — NOT InstanceNorm2d. This is explicitly stated in paper §III-A.
2. **No Spectral Normalization** — the paper does not mention it.
3. **VGG layer l=3 only** in style loss — do NOT use multiple VGG layers.
4. **L_text applies to both cycles** — (I_C, I''_C) AND (I_N, I''_N).
5. **L_region applies to G_E only** — not to G_A branch.
6. **`.detach()`** generator outputs before feeding to discriminator loss.
7. **Patch threshold >100 px** for cracked domain assignment.
8. **Cosine annealing** scheduler — NOT linear decay.
9. **100 epochs** — NOT 300.
10. **LR_G=1e-4, LR_D=4e-4** — NOT 0.0002 for both.
11. Device agnostic: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
12. Seed: `torch.manual_seed(42)`, `np.random.seed(42)` for reproducibility.
13. No placeholder functions — every function must be fully implemented.

---

## 9. Validation Checklist
Before finishing, verify:
- [ ] Generator uses `BatchNorm2d` (grep for `InstanceNorm` — must return zero matches)
- [ ] `D_N.forward()` returns `(output, feat)` tuple with feat from layer index 4
- [ ] VGG is **VGG-16**, single layer at index 3 (`features[:4]`), frozen (`requires_grad=False`)
- [ ] Style loss computed from **one** feature map, NOT a sum across multiple VGG layers
- [ ] `L_text` called for **both** `(I_C, I''_C)` and `(I_N, I''_N)` in train loop
- [ ] `L_region` called **once** (G_E branch only) in train loop
- [ ] M_attn uses `D_N(I_N)` as I_real — **NOT** `D_N(I_C)` — confirm variable name in code
- [ ] Sobel in `L_region` implemented via `F.conv2d` with fixed kernels (differentiable)
- [ ] `.detach()` present on G outputs when computing `loss_D`
- [ ] Patch filter uses `> 100` crack pixels threshold (not `> 0`)
- [ ] `lambda_text=2.0`, `lambda_region=1.0` in config
- [ ] `LR_G=1e-4`, `LR_D=4e-4` in config
- [ ] Scheduler is `CosineAnnealingLR` (not `LambdaLR`)
- [ ] `infer.py` has NO morphological closing (unless explicitly marked `# [not in paper]`)
- [ ] `mBF1` computed at d ∈ {1, 3, 5} using morphological gradient for boundary extraction
- [ ] Any engineering addition (AMP, grad clip, style scaling) marked with `# [not in paper]`

---

## 10. Quick Start

```bash
# Prepare patches from dataset
python dataset.py --data_root ./data --dataset crack500

# Train
python train.py --data_root ./data --dataset crack500 --epochs 100

# Inference on a single image
python infer.py --checkpoint ./checkpoints/epoch_100.pth \
                --image ./test.jpg --output ./result.png

# Evaluate on test set
python evaluate.py --checkpoint ./checkpoints/epoch_100.pth \
                   --data_root ./data --dataset crack500
```