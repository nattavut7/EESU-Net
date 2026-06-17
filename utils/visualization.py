import matplotlib.pyplot as plt


def plot_slice(image, mask=None, slice_idx=None, title=""):
    if slice_idx is None:
        slice_idx = image.shape[-1] // 2
    plt.figure(figsize=(5, 5))
    plt.imshow(image[:, :, slice_idx], cmap="gray")
    if mask is not None:
        plt.imshow(mask[:, :, slice_idx], alpha=0.4)
    plt.title(title)
    plt.axis("off")
    plt.show()
