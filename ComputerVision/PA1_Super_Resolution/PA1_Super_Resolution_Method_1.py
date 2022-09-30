# Method 1

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


def descent_gradient(img_high, img_low, height, width, max_iter):
    for t in range(max_iter):
        img_dh = bilinear(img_high, dsize=(height // 4, width // 4))
        grad = np.subtract(img_dh, img_low)
        grad = bilinear(grad, dsize=(height, width))
        img_high = np.subtract(img_high, np.dot(lr, grad))
        loss = np.sum(
            np.square(
                np.subtract(
                    img_low, bilinear(img_high, dsize=(height // 4, width // 4))
                )
            )
        ).mean()

        if t % 100 == 0:
            print(f"loss at {t} iter: {loss}")

    return img_high


if __name__ == "__main__":
    max_iter = 1_000
    lr = 0.1

    # load upsampled image
    img_high = read_image(dir_path + "upsampled.png")
    height, width = img_high.shape

    # load ground_truth image and resize
    img_gt = read_image(dir_path + "HR.png")
    img_low = bilinear(img_gt, dsize=(height // 4, width // 4))
    # print(img_low.shape)

    result1 = descent_gradient(img_high, img_low, height, width, max_iter)
    cv2.imwrite(dir_path + "results/method1.png")
