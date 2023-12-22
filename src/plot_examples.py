import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line

from src.helpers import _make_box_pts
from src.helpers import make_data


fig, ax = plt.subplots(1, 3, figsize=(12, 4))


def plot(ax: plt.Axes, img: np.ndarray, label: np.ndarray, title: str):
    """Create plot examples for visualizing data.

    Args:
        ax (plt.Axes): Axes to plot.
        img (np.ndarray): Numpy image array.
        label (np.ndarray): Location and angle of the spaceship.
        title (str): Title string indicator.
    """
    ax.imshow(img, cmap="gray")
    ax.set_title(title)

    if label.size > 0:
        x, y, _, _, _ = label
        ax.scatter(x, y, c="r")

        xy = _make_box_pts(*label)
        ax.plot(xy[:, 0], xy[:, 1], c="r")


img, label = make_data(has_spaceship=True)
plot(ax[0], img, label, "example (with spaceship)")

img, label = make_data(has_spaceship=False)
plot(ax[1], img, label, "example (without spaceship)")


img, label = make_data(has_spaceship=True, no_lines=0, noise_level=0)
plot(ax[2], img, label, "example (without noise)")

# Generate an example image
# fig.savefig("example.png")
