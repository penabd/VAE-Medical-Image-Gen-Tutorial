 # Move tensors back to CPU for plotting
imgs = imgs.cpu()
recon = recon.cpu()

import matplotlib.pyplot as plt

# Plot a few examples
n = 5
plt.figure(figsize=(10, 4))

for i in range(n):
    # Original
    plt.subplot(2, n, i + 1)
    plt.imshow(imgs[i][0], cmap="gray")
    plt.axis("off")
    if i == 0:
        plt.title("Original")

    # Reconstruction
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(recon[i][0], cmap="gray")
    plt.axis("off")
    if i == 0:
        plt.title("Reconstruction")

plt.tight_layout()
plt.show()