# Method 2

# import Libraries
import numpy as np
import cv2

dir_path = "./images/"


def read_image(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # cv2.imshow(gray_img)
    return gray_img


def bilinear(img_name, dsize):
    return cv2.resize(img_name, dsize=dsize, interpolation=cv2.INTER_LINEAR)


def apply_sobel(img_high):
    dx = cv2.Sobel(img_high, -1, 1, 0)
    dy = cv2.Sobel(img_high, -1, 0, 1)

    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)
    sobel = abs_dx + abs_dy

    norm = np.linalg.norm(sobel)
    grad_dl = sobel / norm + 1e-10

    return grad_dl


def apply_laplacian(img_high):
    laplacian = cv2.Laplacian(img_high, -1)

    norm = np.linalg.norm(laplacian)
    grad_derivate = laplacian / norm

    return grad_derivate, laplacian


def descent_gradient(img_high, img_low, height, width, max_iter):
    for t in range(max_iter):
        laplacian = cv2.Laplacian(img_high, -1)
        img_dh = cv2.resize(img_high, dsize=(height // 4, width // 4))
        grad = np.subtract(img_dh, img_low)
        grad = cv2.resize(grad, dsize=(height, width)) - beta * (
            laplacian - laplacian_grad
        )
        img_high = np.subtract(img_high, np.dot(lr, grad))
        loss = np.sum(
            np.square(
                np.subtract(
                    img_low, cv2.resize(img_high, dsize=(height // 4, width // 4))
                )
            )
        ).mean()

    if t % 100 == 0:
        print(f"loss at {t} iter: {loss}")

    return img_high


if __name__ == "__main__":
    gamma = 6
    beta = 0.001
    max_iter = 1_000
    lr = 0.1

    # load upsampled image
    img_high = read_image(dir_path + "upsampled.png")
    height, width = img_high.shape

    # load ground_truth image and resize
    img_gt = read_image(dir_path + "HR.png")
    img_low = bilinear(img_gt, dsize=(height // 4, width // 4))
    # print(img_low.shape)
    grad_dl = apply_sobel(img_high)
    grad_derivate, laplacian = apply_laplacian(img_high)

    edge = grad_dl - grad_derivate
    edge = np.clip(edge, a_min=0.0, a_max=1.0)

    laplacian_grad = gamma * laplacian * (edge / grad_dl)

    result2 = descent_gradient(img_high, img_low, height, width, max_iter)
    cv2.imwrite(dir_path + "results/method2.png", result2)
