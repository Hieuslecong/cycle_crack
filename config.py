# config.py — Cycle-Crack Hyperparameters (v3, paper-accurate)

# ── Architecture ──
IMG_SIZE      = 256
IN_CHANNELS   = 3
OUT_CHANNELS  = 3
NGF           = 64          # generator base filters
NDF           = 64          # discriminator base filters
N_ENC_LAYERS  = 8           # encoder depth (UNet-256)

# ── Training [paper §IV-B] ──
BATCH_SIZE    = 16
EPOCHS        = 100
LR_G          = 1e-4        # generator learning rate
LR_D          = 4e-4        # discriminator learning rate
BETA1         = 0.5
BETA2         = 0.999
SCHEDULER     = "cosine"    # CosineAnnealingLR for both G and D

# ── Loss weights [paper Eq.2 and §IV-B] ──
LAMBDA_IDT    = 0.5
LAMBDA_CYCLE  = 10.0
LAMBDA_ADV    = 1.0
LAMBDA_TEXT   = 2.0         # 2.0, NOT 5.0
LAMBDA_REGION = 1.0         # 1.0, NOT 10.0

# ── Identity sub-weights [paper Eq.3] ──
LAMBDA_GE     = 1.0
LAMBDA_GA     = 1.0

# ── Cycle sub-weights [paper Eq.5] ──
LAMBDA_C      = 1.0         # crack-elimination cycle branch
LAMBDA_N      = 1.0         # crack-addition cycle branch

# ── Dataset [paper §IV-A] ──
CRACK_PIXEL_THRESHOLD = 100   # patch must have >100 crack pixels
STRIDE_CRACK500       = 256
STRIDE_DEEPCRACK      = 128
STRIDE_CFD            = 128

# ── Inference [paper §III-B] ──
BILATERAL_D           = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# ── Paths ──
CRACK_DIR     = "data/crack/"
NORMAL_DIR    = "data/noncrack/"
SAVE_DIR      = "checkpoints/"
NUM_WORKERS   = 4
