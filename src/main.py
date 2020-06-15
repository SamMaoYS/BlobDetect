import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def gaussian_kernel(g_sigma, g_size):
    if g_size % 2 == 0:
        g_size += 1
    idx_range = np.linspace(-(g_size - 1) / 2., (g_size - 1) / 2., g_size)
    x_idx, y_idx = np.meshgrid(idx_range, idx_range)
    tmp_cal = 1. / (2. * g_sigma ** 2)
    kernel = np.exp(-(np.square(x_idx) + np.square(y_idx)) * tmp_cal)
    kernel /= np.sum(kernel)
    return kernel


def main():
    img = cv2.imread("../data/butterfly.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    sigma = 2
    size = 31
    kernel = gaussian_kernel(sigma, size)
    print(kernel)

    idx_range = np.linspace(-(99 - 1) / 2., (99 - 1) / 2., 99)
    x, y = np.meshgrid(idx_range, idx_range)
    z = np.pad(kernel, ((49-15, 49-15), (49-15, 49-15)), 'constant')

    fig = plt.figure(figsize=(12, 6))
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, -z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none', antialiased=True)

    ax.set_zlim(-0.1, 0)
    ax.set_zticks(np.linspace(-0.1, 0, 5))
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
