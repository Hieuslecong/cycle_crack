"""train_v3.py — Cycle-Crack Training Engine (v3.1, Professional Refactor)

Features:
    - Modular Trainer class
    - Visual sample logging to TensorBoard (monitoring progress by images)
    - Paper-accurate loss weights & architectures
    - Automatic mixed precision (AMP) for speed
"""
import os, sys, argparse, itertools, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import config as C
from model import UNet256, PatchGAN70, init_weights
from loss  import LSGANLoss, CycleLoss, IdentityLoss, TextureLoss, RegionLoss
from data.dataset import UnpairedCrackDataset
from data.transforms import get_transforms

class CycleCrackTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Models
        self.G_E = init_weights(UNet256(C.IN_CHANNELS, C.OUT_CHANNELS, C.NGF).to(self.device))
        self.G_A = init_weights(UNet256(C.IN_CHANNELS, C.OUT_CHANNELS, C.NGF).to(self.device))
        self.D_N = init_weights(PatchGAN70(C.IN_CHANNELS, C.NDF).to(self.device))
        self.D_C = init_weights(PatchGAN70(C.IN_CHANNELS, C.NDF).to(self.device))
        
        # 2. Dataset
        tf = get_transforms(image_size=C.IMG_SIZE, is_train=True)
        crack_dir = os.path.join(args.data_root, C.CRACK_DIR)
        normal_dir = os.path.join(args.data_root, C.NORMAL_DIR)
        self.ds = UnpairedCrackDataset(crack_dir, normal_dir, transform=tf)
        self.dl = DataLoader(self.ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=C.NUM_WORKERS, drop_last=True, pin_memory=True)
        
        # 3. Optimizers
        self.opt_G = torch.optim.Adam(itertools.chain(self.G_E.parameters(), self.G_A.parameters()),
                                      lr=C.LR_G, betas=(C.BETA1, C.BETA2))
        self.opt_D = torch.optim.Adam(itertools.chain(self.D_N.parameters(), self.D_C.parameters()),
                                      lr=C.LR_D, betas=(C.BETA1, C.BETA2))
        
        # 4. Schedulers
        self.sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_G, T_max=args.epochs)
        self.sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_D, T_max=args.epochs)
        
        # 5. Losses
        self.crit_adv    = LSGANLoss().to(self.device)
        self.crit_cycle  = CycleLoss().to(self.device)
        self.crit_idt    = IdentityLoss().to(self.device)
        self.crit_text   = TextureLoss().to(self.device)
        self.crit_region = RegionLoss().to(self.device)
        
        # 6. Tools
        self.writer = SummaryWriter(log_dir=os.path.join(C.SAVE_DIR, 'logs'))
        self.scaler_G = torch.cuda.amp.GradScaler()
        self.scaler_D = torch.cuda.amp.GradScaler()
        self.step = 0
        self.start_epoch = 0

        self._resume_checkpoint()

    def _resume_checkpoint(self):
        ckpt_path = os.path.join(C.SAVE_DIR, 'GE_latest.pth')
        if os.path.exists(ckpt_path):
            try:
                self.G_E.load_state_dict(torch.load(os.path.join(C.SAVE_DIR, 'GE_latest.pth'), map_location=self.device))
                self.G_A.load_state_dict(torch.load(os.path.join(C.SAVE_DIR, 'GA_latest.pth'), map_location=self.device))
                self.D_N.load_state_dict(torch.load(os.path.join(C.SAVE_DIR, 'DN_latest.pth'), map_location=self.device))
                self.D_C.load_state_dict(torch.load(os.path.join(C.SAVE_DIR, 'DC_latest.pth'), map_location=self.device))
                print("==> Resumed from latest checkpoint.")
            except Exception as e:
                print(f"==> [Warning] Resume failed: {e}. Starting fresh.")

    @torch.no_grad()
    def log_visuals(self, batch, epoch):
        self.G_E.eval(); self.G_A.eval()
        I_C = batch['crack'][:4].to(self.device)
        I_N = batch['normal'][:4].to(self.device)
        
        # Generate
        fake_N = self.G_E(I_C)
        rec_C  = self.G_A(fake_N)
        fake_C = self.G_A(I_N)
        rec_N  = self.G_E(fake_C)
        
        # Create grids
        def denorm(x): return (x + 1) / 2
        grid_C = make_grid(torch.cat([I_C, fake_N, rec_C], dim=0), nrow=4)
        grid_N = make_grid(torch.cat([I_N, fake_C, rec_N], dim=0), nrow=4)
        
        self.writer.add_image('Visuals/Crack_to_Normal_to_Rec', denorm(grid_C), epoch)
        self.writer.add_image('Visuals/Normal_to_Crack_to_Rec', denorm(grid_N), epoch)
        self.G_E.train(); self.G_A.train()

    def train_epoch(self, epoch):
        self.G_E.train(); self.G_A.train(); self.D_N.train(); self.D_C.train()
        epoch_start = time.time()
        
        for i, batch in enumerate(self.dl):
            I_C = batch['crack'].to(self.device)
            I_N = batch['normal'].to(self.device)

            # --- Forward Pass ---
            with torch.cuda.amp.autocast():
                # Domain C -> N -> C
                fake_N = self.G_E(I_C)
                rec_C  = self.G_A(fake_N)
                # Domain N -> C -> N
                fake_C = self.G_A(I_N)
                rec_N  = self.G_E(fake_C)
                # Identity
                # Computed using fake_N and fake_C directly [paper Eq.3]

            # --- Step 1: Optimize Discriminators ---
            self.opt_D.zero_grad()
            with torch.cuda.amp.autocast():
                # D_N: predicts real Normal vs Fake Normal (from C)
                p_real_N, _ = self.D_N(I_N)
                p_fake_N, _ = self.D_N(fake_N.detach())
                l_DN = self.crit_adv(p_real_N, True) + self.crit_adv(p_fake_N, False)
                # D_C: predicts real Crack vs Fake Crack (from N)
                p_real_C, _ = self.D_C(I_C)
                p_fake_C, _ = self.D_C(fake_C.detach())
                l_DC = self.crit_adv(p_real_C, True) + self.crit_adv(p_fake_C, False)
                loss_D = l_DN + l_DC
            
            self.scaler_D.scale(loss_D).backward()
            self.scaler_D.step(self.opt_D)
            self.scaler_D.update()

            # --- Step 2: Optimize Generators ---
            self.opt_G.zero_grad()
            with torch.cuda.amp.autocast():
                # Adversarial
                pg_fake_N, _ = self.D_N(fake_N)
                pg_fake_C, _ = self.D_C(fake_C)
                l_adv = self.crit_adv(pg_fake_N, True) + self.crit_adv(pg_fake_C, True)
                
                # Cycle
                l_cycle = C.LAMBDA_C * self.crit_cycle(rec_C, I_C) + \
                          C.LAMBDA_N * self.crit_cycle(rec_N, I_N)
                
                # Identity
                l_idt = C.LAMBDA_GE * self.crit_idt(fake_N, I_N) + \
                        C.LAMBDA_GA * self.crit_idt(fake_C, I_C)
                
                # Region Consistency (G_E branch only)
                _, f_real_N = self.D_N(I_N)
                _, f_fake_N = self.D_N(fake_N)
                l_region = self.crit_region(fake_N, I_C, f_real_N.detach(), f_fake_N.detach())

            # Texture loss (fp32 to prevent VGG overflow)
            with torch.cuda.amp.autocast(enabled=False):
                l_text = self.crit_text(rec_C.float(), I_C.float()) + \
                         self.crit_text(rec_N.float(), I_N.float())

                loss_G = C.LAMBDA_ADV * l_adv.float() + \
                         C.LAMBDA_CYCLE * l_cycle.float() + \
                         C.LAMBDA_IDT * l_idt.float() + \
                         C.LAMBDA_TEXT * l_text + \
                         C.LAMBDA_REGION * l_region.float()

            self.scaler_G.scale(loss_G).backward()
            self.scaler_G.step(self.opt_G)
            self.scaler_G.update()

            # Logging
            if self.step % 10 == 0:
                self.writer.add_scalar('Loss/G_Total', loss_G.item(), self.step)
                self.writer.add_scalar('Loss/D_Total', loss_D.item(), self.step)
                self.writer.add_scalar('Loss/Adv', l_adv.item(), self.step)
                self.writer.add_scalar('Loss/Cycle', l_cycle.item(), self.step)
                self.writer.add_scalar('Loss/Region', l_region.item(), self.step)
            
            self.step += 1
            if i % 20 == 0:
                print(f"Epoch {epoch+1} [{i}/{len(self.dl)}] L_G: {loss_G.item():.4f} L_D: {loss_D.item():.4f}", end='\r')

        print(f"\n>> Epoch {epoch+1} finished in {time.time()-epoch_start:.2f}s")
        self.log_visuals(batch, epoch)
        self.sched_G.step(); self.sched_D.step()

    def run(self):
        print(f"Starting training for {self.args.epochs} epochs on {self.device}...")
        for epoch in range(self.start_epoch, self.args.epochs):
            self.train_epoch(epoch)
            
            # Save Checkpoints
            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.args.epochs:
                self.save_model(epoch + 1)
        
        self.writer.close()

    def save_model(self, epoch):
        os.makedirs(C.SAVE_DIR, exist_ok=True)
        for name, net in [('GE', self.G_E), ('GA', self.G_A), ('DN', self.D_N), ('DC', self.D_C)]:
            torch.save(net.state_dict(), os.path.join(C.SAVE_DIR, f'{name}_latest.pth'))
            torch.save(net.state_dict(), os.path.join(C.SAVE_DIR, f'{name}_epoch_{epoch}.pth'))
        print(f"==> Checkpoint saved at epoch {epoch}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',  type=str, default='.')
    p.add_argument('--epochs',     type=int, default=C.EPOCHS)
    p.add_argument('--batch_size', type=int, default=C.BATCH_SIZE)
    args = p.parse_args()

    trainer = CycleCrackTrainer(args)
    trainer.run()

if __name__ == '__main__':
    main()
