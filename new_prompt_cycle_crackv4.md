# System Instruction: Cycle-Crack Implementation Agent (v3 — Paper-Accurate + Verified)

## Role
You are an expert PyTorch engineer. Implement the **Cycle-Crack** framework **exactly** as described in the paper:
> *"Cycle-Crack: Weakly Supervised Pavement Crack Segmentation via Dual-Cycle Generative Adversarial Learning"*
> Hongliang Yang, Cong Zhang, Lixin Zhang, Hongmin Liu, Bin Fan — IEEE TIM-25-14285

Every implementation decision must trace back to the paper. Where the paper is silent, use the most conservative standard choice and mark it with `# [not in paper — reason]`.

**Verified against:** Full PDF text extraction + cross-check of all equations (Eq.1–13) and §III-A, §III-B, §III-C, §IV-A, §IV-B.

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

## 1. `config.py` — Hyperparameters

All values below are taken **directly from paper §IV-B** unless marked `[not in paper]`.

```python
# ── Architecture ──────────────────────────────────────────
IMG_SIZE      = 256          # [paper §III-A: "Unet-256", input 256×256]
IN_CHANNELS   = 3            # [paper §III-A: RGB images]
OUT_CHANNELS  = 3            # [paper §III-A: tanh output]
NGF           = 64           # [paper §III-A: first encoder channel = 64]
NDF           = 64           # [paper §III-A: first discriminator channel = 64]
N_ENC_LAYERS  = 8            # [paper §III-A: "eight downsampling layers"]

# ── Training ──────────────────────────────────────────────
BATCH_SIZE    = 16           # [paper §IV-B: "batch size (16)"]
EPOCHS        = 100          # [paper §IV-B: "100 training epochs"]
LR_G          = 1e-4         # [paper §IV-B: "1×10^-4 for the generator"]
LR_D          = 4e-4         # [paper §IV-B: "4×10^-4 for the discriminator"]
BETA1         = 0.5          # [paper §IV-B: "β1=0.5"]
BETA2         = 0.999        # [paper §IV-B: "β2=0.999"]
SCHEDULER     = "cosine"     # [paper §IV-B: "decayed via cosine annealing"]

# ── Loss weights ─────────────────────────────────────────
# All from paper §IV-B: "λ_idt=0.5, λ_cycle=10.0, λ_adv=1.0, λ_text=2.0, λ_region=1.0"
LAMBDA_IDT    = 0.5
LAMBDA_CYCLE  = 10.0
LAMBDA_ADV    = 1.0
LAMBDA_TEXT   = 2.0
LAMBDA_REGION = 1.0

# ── Identity sub-weights [paper Eq.3] ─────────────────────
LAMBDA_GE     = 1.0          # [not in paper — paper writes λ_GE and λ_GA but gives no numeric value; default=1.0]
LAMBDA_GA     = 1.0          # [not in paper — same as above]

# ── Cycle sub-weights [paper Eq.5] ───────────────────────
LAMBDA_C      = 1.0          # [not in paper — paper writes λ_C and λ_N but gives no numeric value; default=1.0]
LAMBDA_N      = 1.0          # [not in paper — same as above]

# ── Dataset ───────────────────────────────────────────────
CRACK_PIXEL_THRESHOLD = 100  # [paper §IV-A: "those with over 100 crack pixels to the cracked subset"]
STRIDE_CRACK500       = 256  # [paper §IV-A: "stride is set to 256 pixels for the Crack500 dataset"]
STRIDE_DEEPCRACK      = 128  # [paper §IV-A: "128 pixels for the DeepCrack dataset"]
STRIDE_CFD            = 128  # [paper §IV-A: "128 pixels for... the CFD dataset"]

# ── Inference ─────────────────────────────────────────────
BILATERAL_D           = 9    # [paper §III-B: "bilateral filter B(·)"; params not specified in paper]
BILATERAL_SIGMA_COLOR = 75   # [not in paper — standard bilateral filter parameters]
BILATERAL_SIGMA_SPACE = 75   # [not in paper — standard bilateral filter parameters]
```

---

## 2. `model.py` — Architecture

### 2.1 Generator: `UNet256`

**Source: paper §III-A** — *"Both generators G_E and G_A adopt a U-Net–based encoder–decoder architecture (Unet-256) with skip connections."*

**CRITICAL**: Paper explicitly states **BatchNorm2d** — *"Batch Normalization"* — NOT InstanceNorm2d.

```
Encoder (8 downsampling blocks)  [paper §III-A]:
  Each block: Conv2d(4×4, stride=2, padding=1) → BatchNorm2d → LeakyReLU(α=0.2)
  Channel progression: {64, 128, 256, 512, 512, 512, 512, 512}
  First encoder block: NO BatchNorm  [not in paper — following pix2pix/CycleGAN convention]

Decoder (8 upsampling blocks)  [paper §III-A]:
  Each block: ConvTranspose2d(4×4, stride=2, padding=1) → BatchNorm2d → ReLU
  Channel progression after skip-concat: mirrors encoder
  Dropout(p=0.5) in "intermediate layers"  [paper §III-A: exact count not specified]
  # [not in paper — following pix2pix convention: Dropout on 3 innermost decoder layers]
  Skip connections: concat encoder[i] with decoder[8-i] along channel dim

Final layer: Conv2d(1×1) → Tanh  →  output ∈ [-1, 1]  [paper §III-A]
```

Instantiate **two** generators [paper §III-A]:
- `G_E`: crack elimination — translates cracked domain C → normal domain N
- `G_A`: crack addition — translates normal domain N → cracked domain C

Both share the same `UNet256` class with identical hyperparameters.

```python
class UNet256(nn.Module):
    """
    UNet-256 generator as described in paper §III-A.
    - BatchNorm2d (NOT InstanceNorm2d) — paper explicitly states "Batch Normalization"
    - Skip connections via concatenation along channel dim
    - Dropout(0.5) in intermediate (innermost) decoder layers
      [not in paper — exact count not specified; following pix2pix: 3 innermost]
    - First encoder block has no BN
      [not in paper — following pix2pix/CycleGAN convention]
    """
```

### 2.2 Discriminator: `PatchGAN70`

**Source: paper §III-A** — *"Both discriminators (D_N, D_C) use a 70×70 PatchGAN with five convolutional layers (4×4 kernels) and channel depths {64, 128, 256, 512, 1}, with LeakyReLU activations."*

```
Layer 1: Conv2d(in_ch, 64,  4, stride=2, pad=1) → LeakyReLU(0.2)        [no BN — paper: first layer no norm]
Layer 2: Conv2d(64,  128,   4, stride=2, pad=1) → BatchNorm2d → LeakyReLU(0.2)
Layer 3: Conv2d(128, 256,   4, stride=2, pad=1) → BatchNorm2d → LeakyReLU(0.2)
Layer 4: Conv2d(256, 512,   4, stride=1, pad=1) → BatchNorm2d → LeakyReLU(0.2)
Layer 5: Conv2d(512,   1,   4, stride=1, pad=1)                           [no BN, no activation]

Strides for layers 1–3: stride=2  [not in paper — following standard 70×70 PatchGAN]
Strides for layers 4–5: stride=1  [not in paper — same]
```

**NOTE**: Paper does NOT mention Spectral Normalization. Do NOT add it.

`forward()` must return **both** the final output AND the feature map from **layer 4** (used by `L_region` Eq.11–12):

```python
def forward(self, x):
    feats = []
    for layer in self.layers:
        x = layer(x)
        feats.append(x)
    # feats[3] = output after layer 4 (0-indexed), shape: (B, 512, H', W')
    return feats[-1], feats[3]   # (patch_pred, intermediate_features)
```

Instantiate **two** discriminators [paper §III-A]:
- `D_N`: judges "is this a real normal (crack-free) image?"
- `D_C`: judges "is this a real cracked image?"

---

## 3. `loss.py` — All 5 Loss Functions

### 3.1 Identity Loss `L_idt` [paper Eq.3]

**Paper Eq.3:**
```
L_idt = λ_GE * ||G_E(I_C) - I_N||_1  +  λ_GA * ||G_A(I_N) - I_C||_1
```

> **Note on Eq.3 semantics**: The paper defines identity loss as passing each domain's image through the *opposite* generator and comparing to the *target* domain image. This differs from the CycleGAN identity convention (pass target image through its own generator). The paper's formulation is used as-is.

```python
def identity_loss(G_E, G_A, I_C, I_N, lambda_GE, lambda_GA):
    loss = lambda_GE * F.l1_loss(G_E(I_C), I_N) \
         + lambda_GA * F.l1_loss(G_A(I_N), I_C)
    return loss
```

### 3.2 Adversarial Loss `L_adv` [paper Eq.4] — LSGAN

**Paper Eq.4 (verbatim):**
```
L_adv = E[(D_N(I_C) - 1)²] + E[D_N(I'_C)²]
       + E[(D_C(I_N) - 1)²] + E[D_C(I'_N)²]
```

> ⚠️ **Paper typo in Eq.4**: The paper writes `D_N(I_C)` and `D_C(I_N)` for the real-image terms, which is logically reversed — D_N judges normal images, so the real image for D_N should be `I_N`, not `I_C`. This is a typographical error in the paper. The **correct logical implementation** is:
>
> ```
> # Correct interpretation (logic, not paper literal):
> L_D = E[(D_N(I_N) - 1)²]  +  E[D_N(G_E(I_C).detach())²]    # D_N: normal domain
>      + E[(D_C(I_C) - 1)²]  +  E[D_C(G_A(I_N).detach())²]    # D_C: cracked domain
>
> # Generator adversarial (fool discriminators):
> L_G_adv = E[(D_N(G_E(I_C)) - 1)²]  +  E[(D_C(G_A(I_N)) - 1)²]
> ```

**CRITICAL**: Use `.detach()` on generator outputs when computing `L_D` to prevent gradients flowing back through G.

```python
def lsgan_loss(pred, is_real):
    target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)

def discriminator_loss(D_N, D_C, I_N, I_C, I_C_fake, I_N_fake):
    # I_C_fake = G_E(I_C).detach()
    # I_N_fake = G_A(I_N).detach()
    pred_real_N, _ = D_N(I_N)
    pred_fake_N, _ = D_N(I_C_fake)   # I_C_fake must be detached
    pred_real_C, _ = D_C(I_C)
    pred_fake_C, _ = D_C(I_N_fake)   # I_N_fake must be detached

    loss = lsgan_loss(pred_real_N, True)  + lsgan_loss(pred_fake_N, False) \
         + lsgan_loss(pred_real_C, True)  + lsgan_loss(pred_fake_C, False)
    return loss

def generator_adv_loss(D_N, D_C, I_C_fake, I_N_fake):
    # I_C_fake = G_E(I_C)  — NO detach here
    # I_N_fake = G_A(I_N)  — NO detach here
    pred_fake_N, _ = D_N(I_C_fake)
    pred_fake_C, _ = D_C(I_N_fake)
    return lsgan_loss(pred_fake_N, True) + lsgan_loss(pred_fake_C, True)
```

### 3.3 Cycle Consistency Loss `L_cycle` [paper Eq.5]

**Paper Eq.5 (verbatim):**
```
L_cycle = λ_C * ||G_A(G_E(I_C)) - I_C||_1  +  λ_N * ||G_E(G_A(I_N) - I_N)||_1
```

> ⚠️ **Paper typo in Eq.5**: The paper writes `G_E(G_A(I_N) - I_N)` — the bracket closes after `I_N`, placing the subtraction **inside** G_E. This is mathematically invalid (cannot subtract images then pass difference as generator input). This is a clear typographical error where the closing parenthesis of G_E is misplaced.
>
> **Correct interpretation:**
> ```
> λ_N * ||G_E(G_A(I_N)) - I_N||_1
> ```
> i.e., apply G_A to I_N, then apply G_E to the result, then compare with original I_N.

```python
def cycle_loss(G_E, G_A, I_C, I_N, I_C_fake, I_N_fake, lambda_C, lambda_N):
    # I_C_fake = G_E(I_C)  — crack → fake normal
    # I_N_fake = G_A(I_N)  — normal → fake cracked
    I_C_rec  = G_A(I_C_fake)   # fake normal → reconstructed crack
    I_N_rec  = G_E(I_N_fake)   # fake cracked → reconstructed normal

    # [paper typo Eq.5 corrected: G_E(G_A(I_N)) - I_N, NOT G_E(G_A(I_N) - I_N)]
    loss = lambda_C * F.l1_loss(I_C_rec, I_C) \
         + lambda_N * F.l1_loss(I_N_rec, I_N)
    return loss, I_C_rec, I_N_rec
```

### 3.4 Texture Consistency Loss `L_text` [paper §III-C, Eq.6–10]

**Applied to BOTH cycle directions** [paper §III-C: *"Except for the region consistency loss, all components are applied to both the crack elimination and crack addition branches."*]

- Cycle C→N→C: pair `(I_C, I''_C)` where `I''_C = G_A(G_E(I_C))`
- Cycle N→C→N: pair `(I_N, I''_N)` where `I''_N = G_E(G_A(I_N))`

Combines **3 sub-terms**:

#### a) MS-SSIM Loss [paper Eq.6–7]

```
L_ms_ssim = (1/S) * Σ_{s=1}^{S} (1 - SSIM(I^real_s, I^rec_s))    [Eq.6]
SSIM(x,y)  = [l(x,y)]^α * [c(x,y)]^β * [s(x,y)]^γ                [Eq.7]
  where α = β = γ = 1   [paper §III-C]
```

> **Scale S**: Paper writes `(1/S) * Σ_s` but **does not specify S numerically**.
> `# [not in paper — S=3 scales chosen following standard MS-SSIM; paper omits this value]`
>
> **SSIM window size**: Paper does not specify.
> `# [not in paper — 11×11 Gaussian window following standard SSIM; paper omits this]`
>
> **Downsampling between scales**: Paper does not specify.
> `# [not in paper — downsample by 2× between scales, following MS-SSIM convention]`

```python
def gaussian_window(size=11, sigma=1.5):
    # [not in paper — window size=11, sigma=1.5 following standard SSIM]
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    return window.unsqueeze(0).unsqueeze(0)  # (1,1,11,11)

def ssim_single(x, y, window):
    # α=β=γ=1 [paper Eq.7]
    C1, C2 = 0.01**2, 0.03**2  # [not in paper — standard SSIM stability constants]
    mu_x  = F.conv2d(x, window, padding=5, groups=1)
    mu_y  = F.conv2d(y, window, padding=5, groups=1)
    mu_x2 = mu_x * mu_x; mu_y2 = mu_y * mu_y; mu_xy = mu_x * mu_y
    sigma_x2  = F.conv2d(x*x, window, padding=5, groups=1) - mu_x2
    sigma_y2  = F.conv2d(y*y, window, padding=5, groups=1) - mu_y2
    sigma_xy  = F.conv2d(x*y, window, padding=5, groups=1) - mu_xy
    l = (2*mu_xy + C1) / (mu_x2 + mu_y2 + C1)        # luminance [paper Eq.7]
    c = (2*sigma_xy + C2) / (sigma_x2 + sigma_y2 + C2) # contrast+structure [paper Eq.7]
    return (l * c).mean()

def ms_ssim_loss(I_real, I_rec, S=3):
    # S=3 [not in paper — paper omits S value]
    window = gaussian_window().to(I_real.device)
    loss = 0.0
    x, y = I_real, I_rec
    for s in range(S):
        # Convert to grayscale for SSIM  [not in paper]
        x_g = x.mean(dim=1, keepdim=True)
        y_g = y.mean(dim=1, keepdim=True)
        loss += 1.0 - ssim_single(x_g, y_g, window)
        if s < S - 1:
            x = F.avg_pool2d(x, 2)  # downsample 2× [not in paper]
            y = F.avg_pool2d(y, 2)
    return loss / S
```

#### b) MSGMS Loss [paper Eq.8]

```
L_msgms = (1/S) * Σ_{s=1}^{S} ||GM(I^real_s) - GM(I^rec_s)||_1    [Eq.8]
GM(x) = sqrt((∂x/∂x)² + (∂x/∂y)²)   [Eq.8 — Sobel gradient magnitude]
```

Implement Sobel via `F.conv2d` with **fixed kernels** (fully differentiable) — paper §III-C specifies Sobel:

```python
def sobel_gradient_magnitude(x):
    """Differentiable Sobel gradient magnitude. [paper Eq.8: GM(x)]"""
    # Fixed Sobel kernels — [not in paper — exact kernel values; standard 3×3 Sobel]
    Kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device)
    Ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],  dtype=x.dtype, device=x.device)
    Kx = Kx.view(1,1,3,3).expand(x.shape[1],-1,-1,-1)  # depthwise
    Ky = Ky.view(1,1,3,3).expand(x.shape[1],-1,-1,-1)
    gx = F.conv2d(x, Kx, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, Ky, padding=1, groups=x.shape[1])
    return torch.sqrt(gx**2 + gy**2 + 1e-8)  # +1e-8 for numerical stability [not in paper]

def msgms_loss(I_real, I_rec, S=3):
    # S=3 [not in paper — paper omits S value]
    loss = 0.0
    x, y = I_real, I_rec
    for s in range(S):
        gm_real = sobel_gradient_magnitude(x)
        gm_rec  = sobel_gradient_magnitude(y)
        loss += F.l1_loss(gm_real, gm_rec)
        if s < S - 1:
            x = F.avg_pool2d(x, 2)  # [not in paper — standard multi-scale downsampling]
            y = F.avg_pool2d(y, 2)
    return loss / S
```

#### c) Style Loss [paper Eq.9–10]

```
L_style = Σ_l ||Gram(VGG_l(I_real)) - Gram(VGG_l(I_rec))||²_F    [Eq.9]
Gram(x) = x·x^T / (C·H·W)                                          [Eq.10]
where l=3 used in this work  [paper §III-C]
```

> **VGG version**: Paper states VGG network, *"l=3 used in this work"*. VGG version is NOT specified.
> `# [not in paper — VGG-16 chosen as the standard pretrained VGG; paper omits VGG version]`
>
> **Layer l=3 indexing**: Paper uses 1-indexed layer 3. For VGG-16 features, layer 3 (1-indexed) = `features[0:4]` = [Conv-ReLU-Conv-ReLU] = relu1_2 output.
> `# [not in paper — interpreting l=3 as features[:4] of VGG-16 (relu1_2), following 0-indexed PyTorch convention]`
>
> **Single layer**: Paper says `l=3` — a **single** layer, NOT a sum over multiple layers.

```python
import torchvision.models as models

def build_vgg_extractor():
    """
    Extract features at layer l=3 of VGG network.
    [not in paper — VGG-16 used; paper does not specify VGG version]
    [not in paper — l=3 interpreted as features[:4] (relu1_2) per 0-indexed PyTorch]
    """
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
    extractor = nn.Sequential(*list(vgg.children())[:4])  # single layer output
    for param in extractor.parameters():
        param.requires_grad = False   # freeze VGG weights [paper: pre-trained VGG]
    return extractor

def gram_matrix(x):
    """Gram(x) = x·x^T / (C·H·W)   [paper Eq.10]"""
    B, C, H, W = x.shape
    feat = x.view(B, C, -1)           # (B, C, H*W)
    gram = torch.bmm(feat, feat.transpose(1, 2)) / (C * H * W)
    return gram

def style_loss(vgg_extractor, I_real, I_rec):
    """
    L_style = ||Gram(VGG_l(I_real)) - Gram(VGG_l(I_rec))||²_F
    Single VGG layer (l=3) only.   [paper Eq.9, l=3]
    """
    # [not in paper — VGG expects ImageNet-normalized input; images are in [-1,1]]
    # Rescale from [-1,1] to [0,1] then normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=I_real.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=I_real.device).view(1,3,1,1)
    def preprocess(x):
        return (x * 0.5 + 0.5 - mean) / std   # [-1,1] → normalized

    feat_real = vgg_extractor(preprocess(I_real))
    feat_rec  = vgg_extractor(preprocess(I_rec))
    return F.mse_loss(gram_matrix(feat_real), gram_matrix(feat_rec))

def texture_loss(vgg_extractor, I_real, I_rec):
    """
    L_text = L_ms_ssim + L_msgms + L_style   [paper §III-C]
    Applied to both (I_C, I''_C) and (I_N, I''_N).  [paper §III-C]
    """
    return ms_ssim_loss(I_real, I_rec) \
         + msgms_loss(I_real, I_rec) \
         + style_loss(vgg_extractor, I_real, I_rec)
```

### 3.5 Region Consistency Loss `L_region` [paper Eq.11–12]

**CRITICAL**: Applied **only** to the G_E (crack elimination) branch — NOT to G_A. [paper §III-C: *"Except for the region consistency loss, all components are applied to both..."*]

**Attention mask M_attn [paper Eq.11]:**
```
M_attn = σ(||F(I_real) - F(I_fake)||)
```
where:
- `F(·)` = intermediate discriminator features (layer 4 output of D_N)
- `I_real` = **I_N** — a real normal/crack-free image fed to D_N
- `I_fake` = **G_E(I_C)** — generated crack-free image from G_E
- `σ(·)` = Sigmoid function

> ⚠️ **Critical variable identity**: `I_real` in Eq.11 is **I_N** (real normal image), NOT I_C (cracked image). Both I_N and G_E(I_C) are passed through **D_N** (the normal-domain discriminator). Feeding I_C as "I_real" would be incorrect.

> **Norm in Eq.11**: Paper writes `||F(I_real) - F(I_fake)||` without subscript. Interpret as L1-like collapse over channel dimension:
> `# [not in paper — norm type not specified in Eq.11; using mean over channel dim before sigmoid]`

> **Upsampling M_attn**: D_N intermediate features have smaller spatial size than 256×256. Upsampling is required for element-wise multiplication in Eq.12:
> `# [not in paper — bilinear upsampling of M_attn to input resolution required by Eq.12]`

**Region loss [paper Eq.12]:**
```
L_region = ||∇G_E(I_C) - ∇I_C||_1  +  [||G_E(I_C) - I_C|| ⊙ (1 - M_attn)]
```
where `∇` = Sobel gradient operator, `⊙` = element-wise multiplication.

> **Norm in second term of Eq.12**: Paper writes `||...||` without subscript after `⊙ (1 - M_attn)`.
> `# [not in paper — norm type not specified for pixel diff term; using L1 (mean reduction)]`

```python
def sobel_operator(x):
    """
    Differentiable Sobel gradient magnitude for L_region Eq.12.
    Uses F.conv2d with fixed kernels — NOT torch.gradient.  [paper §III-C]
    """
    Kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device)
    Ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],  dtype=x.dtype, device=x.device)
    Kx = Kx.view(1,1,3,3).expand(x.shape[1],-1,-1,-1)
    Ky = Ky.view(1,1,3,3).expand(x.shape[1],-1,-1,-1)
    gx = F.conv2d(x, Kx, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, Ky, padding=1, groups=x.shape[1])
    return torch.sqrt(gx**2 + gy**2 + 1e-8)

def region_loss(D_N, I_C, I_N, I_C_fake):
    """
    L_region [paper Eq.11–12].
    Applied to G_E branch ONLY.  [paper §III-C]

    Args:
        I_C_fake: G_E(I_C) — generated crack-free image (NOT detached, gradient flows through G_E)
        I_N:      real normal image — used as I_real for D_N  [paper Eq.11: F(I_real) = F(I_N)]
    """
    # Step 1: Compute attention mask M_attn  [paper Eq.11]
    _, feat_real = D_N(I_N)          # I_real = I_N  [paper: F(I^real)]
    _, feat_fake = D_N(I_C_fake)     # I_fake = G_E(I_C)  [paper: F(I^fake)]

    feat_diff = torch.abs(feat_real - feat_fake)             # (B, C, H', W')
    feat_norm = feat_diff.mean(dim=1, keepdim=True)          # collapse channel dim
    # [not in paper — mean over channels before sigmoid; paper norm type unspecified]
    M_attn    = torch.sigmoid(feat_norm)                     # sigmoid  [paper Eq.11]
    M_attn    = F.interpolate(M_attn, size=I_C.shape[-2:],
                              mode='bilinear', align_corners=False)
    # [not in paper — bilinear upsample to input resolution; required by Eq.12]

    # Step 2: Compute L_region  [paper Eq.12]
    grad_fake = sobel_operator(I_C_fake)    # ∇G_E(I_C)
    grad_real = sobel_operator(I_C)         # ∇I_C

    term1 = F.l1_loss(grad_fake, grad_real)
    # [not in paper — L1 for gradient term; paper uses ||...||_1 subscript for term1]

    pixel_diff = (I_C_fake - I_C) * (1.0 - M_attn)
    term2 = pixel_diff.abs().mean()
    # [not in paper — mean reduction; paper writes ||...|| without subscript for term2]

    return term1 + term2
```

### 3.6 Total Loss [paper Eq.2]

```
L_total = λ_idt * L_idt  +  λ_cycle * L_cycle  +  λ_adv * L_adv
        + λ_text * L_text  +  λ_region * L_region
```

Values: λ_idt=0.5, λ_cycle=10.0, λ_adv=1.0, λ_text=2.0, λ_region=1.0 [paper §IV-B]

---

## 4. `dataset.py` — Data Pipeline

### Sliding Window Patch Extraction [paper §IV-A]

```
window_size = 256 × 256   [paper §IV-A: "fixed window size of 256×256"]
stride:
  Crack500  → 256 px      [paper §IV-A: "256 pixels for the Crack500 dataset"]
  DeepCrack → 128 px      [paper §IV-A: "128 pixels for the DeepCrack dataset"]
  CFD       → 128 px      [paper §IV-A: "128 pixels for... the CFD dataset"]
```

### Patch Assignment [paper §IV-A]

- **Cracked domain**: patch has **> 100 crack pixels** in GT mask
  [paper §IV-A: *"those with over 100 crack pixels to the cracked subset"*]
- **Normal domain**: patch has **0 crack pixels** in GT mask
  [paper §IV-A: *"Patches without crack pixels are assigned to the crack-free subset"*]
- Patches with 1–100 crack pixels are **discarded** as ambiguous
  [paper §IV-A: *"avoiding the inclusion of patches with extremely small or ambiguous crack fragments"*]

### Dataset Statistics [paper §IV-A]

```
Crack500:  4,000 normal | 5,869 cracked total | 1,869 reserved for test
DeepCrack: 435 normal   | 955 cracked total   | 397 reserved for test
CFD:       235 normal   | 384 cracked total   | 84 reserved for test
```

### Dataset Split [paper §IV-A]

- Fixed random seed to avoid overlap between train/test [paper §IV-A]
- Test set = cracked images only
- Training set = all normal images + remaining cracked images

### `CrackDataset` class

```python
class CrackDataset(Dataset):
    """
    Returns unpaired (I_C, I_N) for each training iteration.  [paper §III-A]
    I_N is sampled randomly from the normal pool.
    [not in paper — random sampling strategy not specified; uniform random chosen]

    Augmentation applied to training set only.  [paper §IV-B]
    Normalize to [-1, 1].  [paper §III-A: generator outputs tanh → [-1,1]]
    """
    def __getitem__(self, idx):
        return {'cracked': I_C_tensor, 'normal': I_N_tensor}
```

### Augmentation [paper §IV-B]

Paper states: *"Random scaling, cropping, and rotation are applied for augmentation."*

```python
train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    # [not in paper — ±10° rotation range not specified; conservative choice]
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    # [not in paper — scale range not specified]
    transforms.RandomHorizontalFlip(),
    # [not in paper — horizontal flip not mentioned; added as standard practice]
    transforms.RandomVerticalFlip(),
    # [not in paper — vertical flip not mentioned; added as standard practice]
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # → [-1,1]
])
```

---

## 5. `train.py` — Training Loop

### Optimizer Setup [paper §IV-B]

```python
optimizer_G = Adam(
    list(G_E.parameters()) + list(G_A.parameters()),
    lr=LR_G, betas=(BETA1, BETA2)   # 1e-4, (0.5, 0.999)
)
optimizer_D = Adam(
    list(D_N.parameters()) + list(D_C.parameters()),
    lr=LR_D, betas=(BETA1, BETA2)   # 4e-4, (0.5, 0.999)
)

scheduler_G = CosineAnnealingLR(optimizer_G, T_max=EPOCHS)  # [paper §IV-B: cosine annealing]
scheduler_D = CosineAnnealingLR(optimizer_D, T_max=EPOCHS)
```

### Training Loop Per Iteration

```
─── Forward pass (both cycles) ──────────────────────────────────────────────
I'_C  = G_E(I_C)               # crack → crack-free          [paper §III-B]
I''_C = G_A(I'_C)              # crack-free → crack (cycle 1 reconstruction)
I'_N  = G_A(I_N)               # normal → cracked            [paper §III-B]
I''_N = G_E(I'_N)              # cracked → normal (cycle 2 reconstruction)
I_C_idt = G_E(I_C)             # identity: cracked through G_E  [paper Eq.3]
I_N_idt = G_A(I_N)             # identity: normal through G_A   [paper Eq.3]

─── Step 1: Update Discriminators ──────────────────────────────────────────
# MUST use .detach() to stop gradients flowing into G  [paper: standard GAN training]
pred_real_N, _ = D_N(I_N)
pred_fake_N, _ = D_N(I'_C.detach())      # ← DETACH
pred_real_C, _ = D_C(I_C)
pred_fake_C, _ = D_C(I'_N.detach())      # ← DETACH

loss_D = lsgan(pred_real_N, True) + lsgan(pred_fake_N, False)
       + lsgan(pred_real_C, True) + lsgan(pred_fake_C, False)

optimizer_D.zero_grad()
loss_D.backward()
optimizer_D.step()

─── Step 2: Update Generators ───────────────────────────────────────────────
# Adversarial — NO detach (gradients must flow back to G)
pred_fake_N_G, _ = D_N(I'_C)
pred_fake_C_G, _ = D_C(I'_N)
loss_adv = lsgan(pred_fake_N_G, True) + lsgan(pred_fake_C_G, True)

# Cycle consistency  [paper Eq.5 — typo corrected]
loss_cycle = LAMBDA_C * L1(I''_C, I_C) + LAMBDA_N * L1(I''_N, I_N)

# Identity  [paper Eq.3]
loss_idt = LAMBDA_GE * L1(I_C_idt, I_N) + LAMBDA_GA * L1(I_N_idt, I_C)

# Texture — applied to BOTH cycles  [paper §III-C]
loss_text = texture_loss(vgg, I_C, I''_C) + texture_loss(vgg, I_N, I''_N)

# Region — G_E branch ONLY  [paper §III-C]
# D_N called again with fresh forward to get intermediate features
_, feat_real = D_N(I_N)          # I_real = I_N  [paper Eq.11]
_, feat_fake = D_N(I'_C)         # I_fake = G_E(I_C)
loss_region = region_loss(D_N, I_C, I_N, I'_C)

loss_G = (LAMBDA_ADV    * loss_adv
        + LAMBDA_CYCLE  * loss_cycle
        + LAMBDA_IDT    * loss_idt
        + LAMBDA_TEXT   * loss_text
        + LAMBDA_REGION * loss_region)

optimizer_G.zero_grad()
loss_G.backward()
optimizer_G.step()

─── Schedulers ───────────────────────────────────────────────────────────────
scheduler_G.step()
scheduler_D.step()
```

### Checkpoints & Logging [not in paper]

```python
# [not in paper — checkpoint frequency not specified; saving every 10 epochs]
if epoch % 10 == 0:
    torch.save({
        'epoch': epoch,
        'G_E': G_E.state_dict(), 'G_A': G_A.state_dict(),
        'D_N': D_N.state_dict(), 'D_C': D_C.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D': optimizer_D.state_dict(),
    }, f'checkpoints/epoch_{epoch}.pth')

# Log per epoch: loss_D, loss_G, loss_adv, loss_cycle, loss_idt, loss_text, loss_region
```

### Reproducibility [not in paper]

```python
# [not in paper — fixed seeds for reproducibility]
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

---

## 6. `infer.py` — Inference Pipeline [paper §III-B, Eq.1]

**Paper Eq.1:**
```
M = O(B(|I_C - I'_C|))
```
where B = bilateral filter, O = OTSU threshold [paper §III-B].

```
For a full-resolution input image I_C:

1. Partition into overlapping 256×256 patches
   [paper §III-B + §H: "partitioned into overlapping 256×256 patches"]
   stride = 128 for inference-time overlap  [not in paper — stride not specified for inference]

2. For each patch p:
   a. Normalize to [-1, 1]
   b. p' = G_E(p)                    # crack-free reconstruction  [paper §III-B]
   c. E = |p - p'|                   # error map (3-channel)      [paper Eq.1: |I_C - I'_C|]
   d. E_gray = E.mean(dim=0)         # collapse to 1 channel
      [not in paper — channel collapse method not specified]

3. Stitch patches back to full resolution
   Average overlapping regions  [not in paper — stitching method not specified]

4. Rescale E_gray to [0, 255] uint8  [not in paper — required for cv2 operations]

5. Apply bilateral filter B(·)  [paper §III-B, Eq.1]:
   cv2.bilateralFilter(E_gray_uint8, d=9, sigmaColor=75, sigmaSpace=75)
   [d, sigmaColor, sigmaSpace not specified in paper]

6. Apply OTSU threshold O(·)  [paper §III-B, Eq.1]:
   _, M = cv2.threshold(E_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

7. Return binary mask M  (255=crack, 0=background)
```

**NOTE**: Paper does NOT mention morphological operations in the inference pipeline. Do NOT add morphological closing or opening unless explicitly marked `# [not in paper]`.

---

## 7. `evaluate.py` — Evaluation Metrics [paper §IV-B]

### Pixel-level Metrics [paper §IV-B]

```python
def compute_metrics(pred, gt):
    """
    pred, gt: binary numpy arrays (0/1 or 0/255)
    Returns: Precision, Recall, F1, IoU
    """
    pred = (pred > 0).astype(bool)
    gt   = (gt   > 0).astype(bool)
    TP = (pred & gt).sum()
    FP = (pred & ~gt).sum()
    FN = (~pred & gt).sum()

    P   = TP / (TP + FP + 1e-8)              # [paper §IV-B]
    R   = TP / (TP + FN + 1e-8)              # [paper §IV-B]
    F1  = 2 * P * R / (P + R + 1e-8)        # [paper §IV-B]
    IoU = TP / (TP + FP + FN + 1e-8)        # [paper §IV-B]
    return P, R, F1, IoU
```

### Boundary F1-score (BF1) [paper §IV-B, Eq.13]

**Paper Eq.13:**
```
BF1(d) = 2 × Pb(d) × Rb(d) / (Pb(d) + Rb(d))

where:
  Pb(d) = TPb(d) / (TPb(d) + FPb(d))   — boundary precision within distance d
  Rb(d) = TPb(d) / (TPb(d) + FNb(d))   — boundary recall within distance d
```

Boundaries extracted by **morphological gradient** [paper §IV-B].
`mBF1 = mean BF1 at d ∈ {1, 3, 5}` [paper §IV-B].

```python
from scipy.ndimage import binary_dilation
import cv2

def extract_boundary(mask):
    """Extract boundary using morphological gradient.  [paper §IV-B]"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated  = cv2.dilate(mask.astype(np.uint8), kernel)
    eroded   = cv2.erode(mask.astype(np.uint8), kernel)
    return (dilated - eroded).astype(bool)

def bf1_at_tolerance(pred, gt, d):
    """BF1(d) as defined in paper Eq.13."""
    pred_b = extract_boundary(pred)
    gt_b   = extract_boundary(gt)

    # Dilate GT boundary by d pixels for tolerance matching
    gt_dilated   = binary_dilation(gt_b,   iterations=d)
    pred_dilated = binary_dilation(pred_b, iterations=d)

    TPb = (pred_b & gt_dilated).sum()
    FPb = (pred_b & ~gt_dilated).sum()
    FNb = (gt_b   & ~pred_dilated).sum()

    Pb = TPb / (TPb + FPb + 1e-8)
    Rb = TPb / (TPb + FNb + 1e-8)
    return 2 * Pb * Rb / (Pb + Rb + 1e-8)

def compute_mbf1(pred, gt):
    """mBF1 = mean BF1 at d ∈ {1, 3, 5}  [paper §IV-B]"""
    return np.mean([bf1_at_tolerance(pred, gt, d) for d in [1, 3, 5]])
```

---

## 8. Implementation Rules (Paper-Verified)

| # | Rule | Source |
|---|------|--------|
| 1 | **BatchNorm2d** in Generator, NOT InstanceNorm2d | Paper §III-A explicit |
| 2 | **No Spectral Normalization** | Not mentioned in paper |
| 3 | VGG single layer **l=3** only | Paper Eq.9: "l=3 used in this work" |
| 4 | **L_text** applies to **both** (I_C,I''_C) and (I_N,I''_N) | Paper §III-C |
| 5 | **L_region** applies to **G_E only** | Paper §III-C |
| 6 | `.detach()` on G outputs when computing `loss_D` | Standard GAN |
| 7 | Patch threshold **> 100** crack pixels for cracked domain | Paper §IV-A |
| 8 | **CosineAnnealingLR** scheduler, NOT linear decay | Paper §IV-B |
| 9 | **100 epochs** | Paper §IV-B |
| 10 | LR_G=**1e-4**, LR_D=**4e-4** | Paper §IV-B |
| 11 | Adversarial loss: **Eq.4 has typo** — D_N judges I_N (real normal), D_C judges I_C (real cracked) | Paper Eq.4 typo corrected |
| 12 | Cycle loss: **Eq.5 has typo** — correct form is `G_E(G_A(I_N)) - I_N` | Paper Eq.5 typo corrected |
| 13 | M_attn: `I_real = I_N` (NOT I_C) fed to D_N | Paper Eq.11 |
| 14 | Sobel via `F.conv2d` with fixed kernels (differentiable) | Paper §III-C |
| 15 | Device agnostic: `torch.device("cuda" if torch.cuda.is_available() else "cpu")` | Best practice |
| 16 | All non-paper choices marked `# [not in paper — reason]` | This spec |

---

## 9. Validation Checklist (Paper-Verified)

Before finishing, verify:

**Architecture:**
- [ ] Generator uses `BatchNorm2d` — grep for `InstanceNorm` — must return zero matches
- [ ] Encoder channels exactly: `[64, 128, 256, 512, 512, 512, 512, 512]`
- [ ] Decoder has 8 layers with skip connections via concatenation
- [ ] First encoder block: NO BatchNorm — marked `# [not in paper]`
- [ ] Dropout in innermost decoder layers — marked `# [not in paper — 3 layers]`
- [ ] Generator final layer: Conv2d(1×1) + Tanh
- [ ] D_N and D_C each have exactly 5 conv layers with channels {64,128,256,512,1}
- [ ] `D_N.forward()` returns `(output, feat)` where `feat` = layer 4 output

**Loss functions:**
- [ ] VGG is **VGG-16**, single output at `features[:4]`, all params frozen
- [ ] Style loss uses **ONE** feature map, NOT sum over multiple VGG layers
- [ ] VGG layer choice marked `# [not in paper — VGG-16, features[:4]]`
- [ ] `L_text` called for **both** `(I_C, I''_C)` AND `(I_N, I''_N)` in train loop
- [ ] `L_region` called **once only** (G_E branch) in train loop
- [ ] M_attn uses `D_N(I_N)` as I_real — variable name confirmed as `I_N` not `I_C`
- [ ] M_attn upsampled to input resolution — marked `# [not in paper]`
- [ ] Sobel in `L_region` uses `F.conv2d` with fixed kernels

**Training:**
- [ ] `.detach()` on `I'_C` and `I'_N` when computing `loss_D`
- [ ] No `.detach()` when computing `loss_G` adversarial terms
- [ ] `lambda_text=2.0`, `lambda_region=1.0`
- [ ] `LR_G=1e-4`, `LR_D=4e-4`
- [ ] Scheduler is `CosineAnnealingLR` (NOT `LambdaLR` or `StepLR`)

**Dataset:**
- [ ] Patch filter uses `> 100` crack pixels (strict inequality, NOT `>= 100`)
- [ ] Normal patches have exactly `== 0` crack pixels
- [ ] Flip augmentation marked `# [not in paper]`
- [ ] Rotation range marked `# [not in paper]`

**Inference:**
- [ ] No morphological operations (unless marked `# [not in paper]`)
- [ ] Error map = pixel-level absolute difference `|I_C - G_E(I_C)|`

**Metrics:**
- [ ] Boundaries extracted by morphological gradient (NOT Canny)
- [ ] `mBF1` computed at d ∈ {1, 3, 5}

**Paper typos documented:**
- [ ] Eq.4 typo noted: `D_N(I_C)` → `D_N(I_N)` in real-image term
- [ ] Eq.5 typo noted: misplaced parenthesis in second cycle term

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

---

## Appendix: Paper Typos and Ambiguities

| Location | Paper text | Issue | Correct interpretation |
|----------|-----------|-------|----------------------|
| Eq.4 | `D_N(I_C)` as real-image term | D_N judges normal images, so real = I_N not I_C | Use `D_N(I_N)` |
| Eq.5 | `G_E(G_A(I_N) - I_N)` | Bracket misplaced; subtracting inside G_E is invalid | `G_E(G_A(I_N)) - I_N` |
| §III-C | `l=3` for VGG | VGG version not specified; 1 vs 0-indexed ambiguous | VGG-16, `features[:4]` (relu1_2) |
| §III-C | `(1/S)*Σ_s` | S value not specified | S=3 scales (standard MS-SSIM) |
| §III-A | "intermediate layers" for Dropout | Number of layers not specified | 3 innermost decoder layers (pix2pix convention) |
| Eq.12 | `||G_E(I_C) - I_C||⊙(1-M_attn)` | Norm type not specified for 2nd term | L1 (mean) |
| Eq.11 | `||F(I_real)-F(I_fake)||` | Norm type not specified | Mean over channel dim before sigmoid |