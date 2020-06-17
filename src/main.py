import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def gaussian_kernel(sigma, size):
    if size % 2 == 0:
        size += 1
    idx_range = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    x_idx, y_idx = np.meshgrid(idx_range, idx_range)
    tmp_cal = -(np.square(x_idx) + np.square(y_idx)) / (2. * sigma ** 2)
    kernel = np.exp(tmp_cal)
    kernel[kernel < np.finfo(float).eps * np.amax(kernel)] = 0
    k_sum = np.sum(kernel)
    if k_sum != 0:
        kernel /= np.sum(kernel)
    return kernel


def log_kernel(sigma, size):
    if size % 2 == 0:
        size += 1
    sigma2 = sigma ** 2
    idx_range = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    x_idx, y_idx = np.meshgrid(idx_range, idx_range)
    tmp_cal = -(np.square(x_idx) + np.square(y_idx)) / (2. * sigma2)
    kernel = np.exp(tmp_cal)
    kernel[kernel < np.finfo(float).eps * np.amax(kernel)] = 0
    k_sum = np.sum(kernel)
    if k_sum != 0:
        kernel /= np.sum(kernel)
    tmp_kernel = np.multiply(kernel, np.square(x_idx) + np.square(y_idx) - 2 * sigma2) / (sigma2 ** 2)
    kernel = tmp_kernel - np.sum(tmp_kernel) / (size ** 2)
    return kernel


def show_kernel(kernel):
    size = np.shape(kernel)[0]
    idx_range = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    x, y = np.meshgrid(idx_range, idx_range)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, kernel, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none', antialiased=True)
    ax.view_init(15, 0)
    plt.show()
    plt.cla()
    plt.clf()
    plt.close(fig)


def non_max_suppression(img):
    h, w = np.shape(img)
    result = np.zeros_like(img)
    # TODO: Improve performance
    for i in range(1, h-2):
        for j in range(1, w-2):
            if img[i, j] > 0:
                window = np.array([img[i-1, j-1], img[i-1, j], img[i-1, j+1],
                                   img[i, j-1], img[i, j], img[i, j+1],
                                   img[i+1, j-1], img[i+1, j], img[i+1, j+1]])
                if np.max(window) == img[i, j]:
                    result[i, j] = img[i, j]

    return result


def main():
    img = cv2.imread("../data/butterfly.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    sigma0 = np.sqrt(2)
    k = np.sqrt(2)
    num_scales = 10
    sigmas = sigma0*np.power(k, np.arange(num_scales))
    for i in range(num_scales):
        size = np.int(2*np.ceil(4*sigmas[i])+1)
        kernel = log_kernel(sigmas[i], size)*np.power(sigmas[i], 2)
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        filtered = pow(filtered, 2)
        filtered = non_max_suppression(filtered)


if __name__ == "__main__":
    main()
