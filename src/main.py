import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# Gaussian kernel generator
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


# Laplacian of Gaussian kernel generator
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


# Plot kernel in 3d
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


def main():
    img_name = "butterfly"
    # img_name = "sunflowers"
    img = cv2.imread("../data/" + img_name + ".jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    cv2.normalize(gray, gray, 1, 0, cv2.NORM_MINMAX)

    sigma0 = np.sqrt(2)
    k = np.sqrt(2)
    num_scales = 15
    sigmas = sigma0 * np.power(k, np.arange(num_scales))
    # apply LoG kernel filtering with scaled kernel size and sigma
    img_stack = None
    for i in range(num_scales):
        size = np.int(2 * np.ceil(4 * sigmas[i]) + 1)
        # with Laplacian response normalization
        kernel = log_kernel(sigmas[i], size) * np.power(sigmas[i], 2)
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        filtered = pow(filtered, 2)
        if i == 0:
            img_stack = filtered
        else:
            img_stack = np.dstack((img_stack, filtered))

    # Maximum response extraction
    scale_space = None
    for i in range(num_scales):
        filtered = cv2.dilate(img_stack[:, :, i], np.ones((3, 3)), cv2.CV_32F, (-1, -1), 1, cv2.BORDER_CONSTANT)
        if i == 0:
            scale_space = filtered
        else:
            scale_space = np.dstack((scale_space, filtered))
    max_stack = np.amax(scale_space, axis=2)
    max_stack = np.repeat(max_stack[:, :, np.newaxis], num_scales, axis=2)
    max_stack = np.multiply((max_stack == scale_space), scale_space)

    radius_vec = None
    x_vec = None
    y_vec = None
    for i in range(num_scales):
        radius = np.sqrt(2) * sigmas[i]
        threshold = 0.01
        # filter out redundant response
        valid = (max_stack[:, :, i] == img_stack[:, :, i]) * img_stack[:, :, i]
        valid[valid <= threshold] = 0
        (x, y) = np.nonzero(valid)
        if i == 1:
            x_vec = x
            y_vec = y
            radius_vec = np.repeat(radius, np.size(x))
        else:
            x_vec = np.concatenate((x_vec, x), axis=None)
            y_vec = np.concatenate((y_vec, y), axis=None)
            tmp_vec = np.repeat(radius, np.size(x))
            radius_vec = np.concatenate((radius_vec, tmp_vec), axis=None)

    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for i in range(np.size(x_vec)):
        cv2.circle(out, (y_vec[i], x_vec[i]), np.int(radius_vec[i]), (0, 0, 255), 1)
    cv2.imshow("Blob Num "+str(np.size(x_vec)), out)
    cv2.imwrite("../result/" + img_name + ".jpg", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
