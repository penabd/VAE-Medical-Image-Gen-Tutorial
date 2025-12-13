import torch
from models.cvae import CVAE
from models.losses import cvae_loss
from models.eval import evaluate

# -------------------------
# Setup
# -------------------------

# Change these to your liking
LATENT_DIM = 32
LABEL_DIM = 1
LR = 1e-3
EPOCHS = 50
CHECKPOINT_PATH = "/content/drive/MyDrive/cvae_best2.pt"
LOAD_EXISTING = False

# use GPUs if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Model
# -------------------------
model = CVAE(latent_dim=LATENT_DIM, label_dim=LABEL_DIM).to(device)

best_val_loss = float("inf")

if LOAD_EXISTING:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print("Loaded existing model")

# -------------------------
# Optimizer
# -------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Training loop
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        recon, mu, logvar = model(imgs, labels)
        loss = cvae_loss(recon, imgs, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    val_loss = evaluate(model, val_loader, device)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
    )

    # Checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print("Saved best model")
