img1 = imread("SfM/Data/sfm03.jpg");
img2 = imread("SfM/Data/sfm04.jpg");

img1_gray = single(rgb2gray(img1));
[img1_kp, img1_des] = vl_sift(img1_gray);

img2_gray = single(rgb2gray(img2));
[img2_kp, img2_des] = vl_sift(img2_gray);

[match, scores] = vl_ubcmatch(img1_des, img2_des);