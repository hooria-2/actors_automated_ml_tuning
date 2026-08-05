import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt


LATENT_DIM  = 64#16     # elbow for recon  # size of the latent vector
BATCH_SIZE  = 256    # VAEs are more sensitive to batch size than standard autoencoders because:
                     # KL term is averaged over batch
                     # posterior statistics depend on batch distribution
                     # So: Bigger batch ≠ always better latent space
EPOCHS      = 200
LR          = 1e-3
BETA        = 1.0#0.10      # weight on KL term  (β-VAE: increase for disentanglement)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
VAE_SAVE_PATH   = "vae_mnist.pth"
FM_SAVE_PATH    = "fm_mnist.pth"

LR_FM       = 1e-5
EPOCHS_FM   = 200


class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu      = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

    def forward(self, x):
        h       = self.net(x.view(x.size(0), -1))   # flatten to 784
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)


    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        z = mu + eps * std   where   eps ~ N(0, I)

        This keeps gradients flowing through mu and log_var while still
        sampling a stochastic z during training.
        """
        std = torch.exp(0.5 * log_var)           # σ
        eps = torch.randn_like(std)              # ε ~ N(0,I)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        x_hat       = self.decoder(z)
        return x_hat, mu, log_var

    @torch.no_grad()
    def evaluate(self, x):
        mu, _ = self.encoder(x)
        x_hat = self.decoder(mu)
        return x_hat, mu


    @torch.no_grad()
    def encode(self, x) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return mu

    @torch.no_grad()
    def decode(self, z) -> torch.Tensor:
        return self.decoder(z)


def elbo_loss(x_hat, x, mu, log_var, beta: float = BETA):

    recon_loss = F.mse_loss(x_hat, x, reduction="sum")/ x.size(0)

    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train(model, loader, optimizer, epoch):
    model.train()
    total, recon_t, kl_t = 0.0, 0.0, 0.0
    
    for batch_idx, (x, _) in enumerate(loader):
        x = x.to(DEVICE)
        optimizer.zero_grad()
        x_hat, mu, log_var = model(x)
        loss, recon, kl    = elbo_loss(x_hat, x, mu, log_var)
        loss.backward()
        optimizer.step()
        total   += loss.item()
        recon_t += recon.item()
        kl_t    += kl.item()

    n = len(loader)
    print(
        f"Epoch {epoch:02d} | "
        f"ELBO {total/n:.2e}  "
        f"Recon {recon_t/n:.2e}  "
        f"KL {kl_t/n:.2e}"
    )
    return total/n, recon_t/n, kl_t/n

def batch_ssim(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    kernel_size = 11
    sigma       = 1.5
    coords      = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
    coords     -= kernel_size // 2
    g           = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g           = g / g.sum()
    kernel_2d   = g.unsqueeze(1) @ g.unsqueeze(0)                  # (11,11)
    kernel_2d   = kernel_2d.unsqueeze(0).unsqueeze(0)               # (1,1,11,11)

    pad = kernel_size // 2
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    def local_stats(img):
        mu  = F.conv2d(img, kernel_2d, padding=pad)
        mu2 = mu * mu
        var = F.conv2d(img * img, kernel_2d, padding=pad) - mu2
        return mu, var

    mu_x,    var_x    = local_stats(x)
    mu_xhat, var_xhat = local_stats(x_hat)
    cov = F.conv2d(x * x_hat, kernel_2d, padding=pad) - mu_x * mu_xhat

    numerator   = (2 * mu_x * mu_xhat + C1) * (2 * cov + C2)
    denominator = (mu_x ** 2 + mu_xhat ** 2 + C1) * (var_x + var_xhat + C2)
    ssim_map    = numerator / denominator                            # (B,1,H,W)
    return ssim_map.mean().item()


def batch_psnr(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    mse = F.mse_loss(x_hat, x, reduction="mean")
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(torch.tensor(1.0) / mse)).item()


@torch.no_grad()
def evaluate(model, loader, epoch):
    model.eval()
    total_elbo = total_recon = total_kl = total_ssim = total_psnr = 0.0
    n = len(loader)

    for x, _ in loader:
        x                  = x.to(DEVICE)
        x_hat, mu, log_var = model(x)
        loss, recon, kl    = elbo_loss(x_hat, x, mu, log_var)

        total_elbo  += loss.item()
        total_recon += recon.item()
        total_kl    += kl.item()
        total_ssim  += batch_ssim(x_hat, x)
        total_psnr  += batch_psnr(x_hat, x)

    print(
        f"          Val | "
        f"ELBO {total_elbo/n:.2e}  "
        f"Recon {total_recon/n:.2e}  "
        f"KL {total_kl/n:.2e}  "
        f"SSIM {total_ssim/n:.2f}  "
        f"PSNR {total_psnr/n:.2f} dB"
    )

    return total_elbo/n, total_recon/n, total_kl/n, total_ssim/n, total_psnr/n

@torch.no_grad()
def encode_test_set(model, test_loader):
    model.eval()
    all_z, all_y = [], []
    for x, y in test_loader:
        z = model.encode(x.to(DEVICE))
        all_z.append(z.cpu())
        all_y.append(y)
    return torch.cat(all_z).numpy(), torch.cat(all_y).numpy()

@torch.no_grad()
def save_reconstructions(model, test_loader, fname="reconstructions.png"):
    model.eval()
    x, _ = next(iter(test_loader))
    x    = x[:16].to(DEVICE)
    x_hat, _, _ = model(x)
    
    comparison = torch.cat([x, x_hat])
    save_image(comparison.cpu(), fname, nrow=16)
    print(f"Saved reconstruction grid → {fname}")

def LearnLatent(model):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_total_list = []
    train_recon_list = []
    train_kl_list = []
    test_total_list = []
    test_recon_list = []
    test_kl_list = []
    test_ssim_list = []
    test_psnr_list = []
    
    print(f"Training VAE (latent_dim={LATENT_DIM}, β={BETA}) on {DEVICE}...\n")
    for epoch in range(1, EPOCHS + 1):
        train_total, train_recon, train_kl = train(model, train_loader, optimizer, epoch)
        train_total_list.append(train_total)
        train_recon_list.append(train_recon)
        train_kl_list.append(train_kl)
        
        test_total, test_recon, test_kl, test_ssim, test_psnr = evaluate(model, test_loader, epoch)
        test_total_list.append(test_total)
        test_recon_list.append(test_recon)
        test_kl_list.append(test_kl)
        test_ssim_list.append(test_ssim)
        test_psnr_list.append(test_psnr)

    torch.save(model, VAE_SAVE_PATH)
    print(f"\nModel saved → {VAE_SAVE_PATH}")
    save_reconstructions(model, test_loader)

    print("\nEncoding test set deterministically (z = mu)...")
    latents, labels = encode_test_set(model, test_loader)
    np.save("latent_vectors.npy", latents)
    np.save("latent_labels.npy",  labels)
    print(f"Saved latent_vectors.npy  shape={latents.shape}")
    print(f"Saved latent_labels.npy   shape={labels.shape}")

    print("\n── Latent vector stats (test set) ──")
    print(f"  Mean  : {latents.mean(axis=0)[:4].round(3)}  ...")
    print(f"  Std   : {latents.std(axis=0)[:4].round(3)}  ...")
    print(f"  Min   : {latents.min():.3f}")
    print(f"  Max   : {latents.max():.3f}")
    print("\nDone learning latents.\n")

    plt.plot(train_total_list, label='train')
    plt.plot(test_total_list, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('Latent_loss_curve.png')

    plt.figure()
    plt.plot(train_recon_list, label='train')
    plt.plot(test_recon_list, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('Latent_Recon_curve.png')

    plt.figure()
    plt.plot(train_kl_list, label='train')
    plt.plot(test_kl_list, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('KL Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('Latent_KL_curve.png')

    plt.figure()
    plt.plot(test_ssim_list, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()
    plt.savefig('Latent_SSIM_curve.png')

    plt.figure()
    plt.plot(test_psnr_list, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()
    plt.savefig('Latent_PSNR_curve.png')

    return model

# Flow matching

def compute_pairwise_sq_dists(x, y):
    """
    x: (N, D)
    y: (M, D)
    returns: (N, M) squared L2 distances
    """
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    y2 = (y ** 2).sum(dim=1, keepdim=True).T
    return x2 + y2 - 2 * (x @ y.T)


def rbf_kernel(x, y, sigma=None):
    dists = compute_pairwise_sq_dists(x, y)

    if sigma is None:
        # median heuristic
        sigma = torch.median(dists).sqrt().detach()
        sigma = sigma + 1e-8

    return torch.exp(-dists / (2 * sigma ** 2))

class VelocityField(nn.Module):
    def __init__(self, dim=16, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        # x: (B, dim), t: (B, 1)
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)

def flow_matching_loss(model, x1, device='cpu'):
    """
    x1: real data batch (B, dim)
    """

    B, D = x1.shape
    device = x1.device

    x0 = torch.randn_like(x1)

    t = torch.rand(B, 1, device=device)

    xt = (1 - t) * x0 + t * x1

    v_target = x1 - x0

    v_pred = model(xt, t)

    loss = F.mse_loss(v_pred, v_target)

    return loss

# @torch.no_grad()
# def sampleFM(model, n_samples=64, dim=16, steps=50, device="cpu"):
#     x = torch.randn(n_samples, dim, device=device)

#     dt = 1.0 / steps

#     for i in range(steps):
#         t = torch.full((n_samples, 1), i / steps, device=device)

#         v = model(x, t)
#         x = x + v * dt

#     return x

@torch.no_grad()
def sampleFM(
    model,
    n_samples=10000,
    dim=16,
    steps=50,
    batch_size=256,
    device="cpu"
):

    model.eval()

    all_samples = []

    for start in range(0, n_samples, batch_size):

        current_bs = min(batch_size, n_samples - start)

        x = torch.randn(current_bs, dim, device=device)

        dt = 1.0 / steps

        for i in range(steps):

            t = torch.full(
                (current_bs, 1),
                i / steps,
                device=device
            )

            v = model(x, t)

            x = x + v * dt
        
        x = Latent_model.decode(x)

        all_samples.append(x.cpu())

    return torch.cat(all_samples, dim=0)

def trainFM():
    
    optimizer_FM = torch.optim.Adam(model_FM.parameters(), lr=LR_FM)
    losses = []
    for epoch in range(EPOCHS_FM):
        loss_ = 0
        for batch_idx, (X1, _) in enumerate(train_loader):
            X1 = X1.to(DEVICE)
            with torch.no_grad():
                x1 = Latent_model.encode(X1)

            loss_FM = flow_matching_loss(model_FM, x1)

            optimizer_FM.zero_grad()
            loss_FM.backward()
            optimizer_FM.step()

            if batch_idx % 500 == 0:
                print(f"epoch {epoch} | loss {loss_FM.item():.4f}")
            loss_ += loss_FM.item()
        
        losses.append(loss_)

    torch.save(model_FM, FM_SAVE_PATH)
    print(f"\nModel saved → {FM_SAVE_PATH}")

    plt.figure()
    plt.plot(losses, label='train')
    plt.xlabel('Epochs')
    plt.ylabel('Flow Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('Flow_loss_curve.png')
    
    return model_FM

def compute_mmd(x_real, x_fake, sigma=None):
    """
    MMD^2 between real and fake samples
    """
    K_xx = rbf_kernel(x_real, x_real, sigma)
    K_yy = rbf_kernel(x_fake, x_fake, sigma)
    K_xy = rbf_kernel(x_real, x_fake, sigma)

    m = x_real.size(0)
    n = x_fake.size(0)

    mmd = (
        K_xx.mean()
        + K_yy.mean()
        - 2 * K_xy.mean()
    )
    return mmd.item()

@torch.no_grad()
def evaluate_mmd(model_FM, Latent_model, test_loader, device, num_samples=20000):
    model_FM.eval()
    Latent_model.eval()

    real_z = []

    for x, _ in test_loader:
        x = x.to(device).reshape(x.size(0), -1)
        # z = Latent_model.encode(x)
        real_z.append(x)

        if sum(t.size(0) for t in real_z) > num_samples:
            break

    real_z = torch.cat(real_z, dim=0)[:num_samples]

    fake_z = sampleFM(
        model_FM,
        n_samples=num_samples,
        dim=LATENT_DIM,
        steps=100,
        device=device
    )

    fig, axes = plt.subplots(1, 10, figsize=(15, 2))

    for i in range(10):
        axes[i].imshow(fake_z[i, 0].cpu(), cmap="gray")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig('GeneratedSamples.png')

    fake_z = fake_z.reshape(fake_z.size(0), -1).to(device)
    
    mmd_score = compute_mmd(real_z, fake_z)

    print(f"\nMMD Score (Original Space): {mmd_score:.2e}")

    return mmd_score

if __name__ == "__main__":

    transform = transforms.ToTensor()

    train_set  = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_set   = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)#,  num_workers=2)#, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)#, num_workers=2)#, pin_memory=True)

    Latent_model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    print("# Parameters in VAE: ", sum(p.numel() for p in Latent_model.parameters() if p.requires_grad))

    Latent_model = LearnLatent(Latent_model)

    print('Training the Flow model:')
    model_FM = VelocityField(dim=LATENT_DIM).to(DEVICE)
    print("# Parameters in Flow Model: ", sum(p.numel() for p in model_FM.parameters() if p.requires_grad))

    model_FM = trainFM()

    print('\nEvaluating the flow model:')
    mmd = evaluate_mmd(model_FM, Latent_model, test_loader, DEVICE, num_samples=10000)

    