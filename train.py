import os
import yaml
import argparse
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp

from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from models.networks import init_net
from losses.adversarial import LSGANLoss
from losses.cycle import CycleLoss
from losses.identity import IdentityLoss
from losses.texture import TextureLoss
from losses.region import RegionLoss
from data.dataset import UnpairedCrackDataset
from data.transforms import get_transforms


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_training():
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(config['save_dir'], exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(config['save_dir'], 'logs'))

    # 1. Data
    transform = get_transforms(image_size=config['image_size'], is_train=True)
    dataset = UnpairedCrackDataset(config['crack_dir'], config['normal_dir'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                            num_workers=config['num_workers'], drop_last=True)

    # 2. Models
    # G_E: Crack → Normal (crack elimination)
    # G_A: Normal → Crack (crack addition)
    G_E = init_net(UNetGenerator(in_channels=3, out_channels=3, ngf=config['ngf']), device=device)
    G_A = init_net(UNetGenerator(in_channels=3, out_channels=3, ngf=config['ngf']), device=device)
    D_N = init_net(PatchGANDiscriminator(in_channels=3, ndf=config['ndf']), device=device)
    D_C = init_net(PatchGANDiscriminator(in_channels=3, ndf=config['ndf']), device=device)

    # Resume from checkpoint if exists
    ckpt_ge = os.path.join(config['save_dir'], 'GE_latest.pth')
    if os.path.exists(ckpt_ge):
        try:
            G_E.load_state_dict(torch.load(ckpt_ge, map_location=device, weights_only=True))
            G_A.load_state_dict(torch.load(os.path.join(config['save_dir'], 'GA_latest.pth'), map_location=device, weights_only=True))
            D_N.load_state_dict(torch.load(os.path.join(config['save_dir'], 'DN_latest.pth'), map_location=device, weights_only=True))
            D_C.load_state_dict(torch.load(os.path.join(config['save_dir'], 'DC_latest.pth'), map_location=device, weights_only=True))
            print("Resumed from latest checkpoints.")
        except RuntimeError as e:
            print(f"[WARNING] Could not load checkpoints (architecture mismatch?): {e}")
            print("[WARNING] Starting training from scratch with random weights.")

    # 3. Optimizers — separate LR for G and D [paper §IV-B]
    opt_G = torch.optim.Adam(
        itertools.chain(G_E.parameters(), G_A.parameters()),
        lr=config['lr_G'], betas=(config['beta1'], config['beta2'])
    )
    opt_D = torch.optim.Adam(
        itertools.chain(D_N.parameters(), D_C.parameters()),
        lr=config['lr_D'], betas=(config['beta1'], config['beta2'])
    )

    # 4. Schedulers — CosineAnnealingLR [paper §IV-B]
    num_epochs = config['num_epochs']
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=num_epochs)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=num_epochs)

    # 5. Losses
    criterion_adv    = LSGANLoss().to(device)
    criterion_cycle  = CycleLoss().to(device)
    criterion_idt    = IdentityLoss().to(device)
    criterion_text   = TextureLoss().to(device)
    criterion_region = RegionLoss().to(device)

    scaler_G = amp.GradScaler()  # [not in paper]: AMP FP16 for speed
    scaler_D = amp.GradScaler()  # [not in paper]: AMP FP16 for speed

    global_step = 0

    for epoch in range(num_epochs):
        G_E.train(); G_A.train(); D_N.train(); D_C.train()

        epoch_loss_G, epoch_loss_D = 0.0, 0.0

        for batch in dataloader:
            I_C = batch['crack'].to(device)   # cracked images
            I_N = batch['normal'].to(device)  # normal (crack-free) images

            # ── Forward: both cycles ──────────────────────────────────────
            with amp.autocast():
                # Cycle C → N → C
                I_C_fake = G_E(I_C)       # crack-free reconstruction of cracked
                I_C_rec  = G_A(I_C_fake)  # re-cracked reconstruction

                # Cycle N → C → N
                I_N_fake = G_A(I_N)       # fake-cracked reconstruction of normal
                I_N_rec  = G_E(I_N_fake)  # re-normal reconstruction

                # Identity
                I_C_idt = G_E(I_N)        # G_E(I_N) should ≈ I_N
                I_N_idt = G_A(I_C)        # G_A(I_C) should ≈ I_C

            # ── Step 1: Update Discriminators ─────────────────────────────
            opt_D.zero_grad()
            with amp.autocast():
                pred_real_N, _ = D_N(I_N)
                pred_fake_N, _ = D_N(I_C_fake.detach())   # MUST detach
                pred_real_C, _ = D_C(I_C)
                pred_fake_C, _ = D_C(I_N_fake.detach())   # MUST detach

                loss_D = (criterion_adv(pred_real_N, True) + criterion_adv(pred_fake_N, False)
                        + criterion_adv(pred_real_C, True) + criterion_adv(pred_fake_C, False))

            scaler_D.scale(loss_D).backward()
            scaler_D.unscale_(opt_D)
            torch.nn.utils.clip_grad_norm_(
                list(D_N.parameters()) + list(D_C.parameters()), max_norm=1.0
            )
            scaler_D.step(opt_D)
            scaler_D.update()

            # ── Step 2: Update Generators ─────────────────────────────────
            opt_G.zero_grad()
            with amp.autocast():
                # Adversarial (fool discriminators — no detach)
                pred_fake_N_G, _ = D_N(I_C_fake)
                pred_fake_C_G, _ = D_C(I_N_fake)
                loss_adv = (criterion_adv(pred_fake_N_G, True)
                          + criterion_adv(pred_fake_C_G, True))

                # Cycle Consistency [paper Eq.5]
                loss_cycle = (criterion_cycle(I_C_rec, I_C)
                            + criterion_cycle(I_N_rec, I_N))

                # Identity [paper Eq.3]
                loss_idt = (criterion_idt(I_C_idt, I_N)
                          + criterion_idt(I_N_idt, I_C))

                # Region Loss — G_E branch only [paper Eq.11-12]
                _, feat_real = D_N(I_N)
                _, feat_fake = D_N(I_C_fake)
                loss_region = criterion_region(I_C_fake, I_C, feat_real.detach(), feat_fake.detach())

            # [not in paper]: TextureLoss computed in fp32 to avoid VGG fp16 overflow.
            # VGG Gram matrix MSE overflows fp16 even after 1e-3 scaling.
            with amp.autocast(enabled=False):  # [not in paper]
                loss_text = (criterion_text(I_C_rec.float(), I_C.float())
                           + criterion_text(I_N_rec.float(), I_N.float()))

                loss_G = (config['lambda_adv']    * loss_adv.float()
                        + config['lambda_cycle']  * loss_cycle.float()
                        + config['lambda_idt']    * loss_idt.float()
                        + config['lambda_text']   * loss_text
                        + config['lambda_region'] * loss_region.float())

            scaler_G.scale(loss_G).backward()
            scaler_G.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(  # [not in paper]: gradient clipping for stability
                list(G_E.parameters()) + list(G_A.parameters()), max_norm=1.0
            )
            scaler_G.step(opt_G)
            scaler_G.update()

            # Logging
            writer.add_scalar('Loss/G',        loss_G.item(),      global_step)
            writer.add_scalar('Loss/D',        loss_D.item(),      global_step)
            writer.add_scalar('Loss/G_adv',    loss_adv.item(),    global_step)
            writer.add_scalar('Loss/G_cycle',  loss_cycle.item(),  global_step)
            writer.add_scalar('Loss/G_idt',    loss_idt.item(),    global_step)
            writer.add_scalar('Loss/G_text',   loss_text.item(),   global_step)
            writer.add_scalar('Loss/G_region', loss_region.item(), global_step)
            global_step += 1

            g_val = loss_G.item() if not torch.isnan(loss_G) else float('nan')
            epoch_loss_G += g_val if not (g_val != g_val) else 0.0
            epoch_loss_D += loss_D.item()

        scheduler_G.step()
        scheduler_D.step()

        n_batches = len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"L_G={epoch_loss_G/n_batches:.4f}  L_D={epoch_loss_D/n_batches:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            torch.save(G_E.state_dict(), os.path.join(config['save_dir'], f'GE_epoch_{epoch+1}.pth'))
            torch.save(G_A.state_dict(), os.path.join(config['save_dir'], f'GA_epoch_{epoch+1}.pth'))
            torch.save(D_N.state_dict(), os.path.join(config['save_dir'], f'DN_epoch_{epoch+1}.pth'))
            torch.save(D_C.state_dict(), os.path.join(config['save_dir'], f'DC_epoch_{epoch+1}.pth'))
            # Also keep a "latest" copy for easy resume
            torch.save(G_E.state_dict(), os.path.join(config['save_dir'], 'GE_latest.pth'))
            torch.save(G_A.state_dict(), os.path.join(config['save_dir'], 'GA_latest.pth'))
            torch.save(D_N.state_dict(), os.path.join(config['save_dir'], 'DN_latest.pth'))
            torch.save(D_C.state_dict(), os.path.join(config['save_dir'], 'DC_latest.pth'))

    writer.close()
    print("Training finished.")


if __name__ == '__main__':
    run_training()
