import numpy as np
import cv2 as cv

import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath("./SfM/Step2/")  # 'calibrated_fivepoint.m'가 위치한 경로

# MAXITER = 100
MAXITER = 1
threshold = 5e-4

# intrinsic parameter
K = np.array([[3451.5, 0.0, 2312.0], [0.0, 3451.5, 1734], [0.0,0.0,1.0]])
K_inv = np.linalg.inv(K)

W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# Step 1: Feature Extraction & Matching
img1 = cv.imread('./SfM/Data/sfm03.jpg')
img2 = cv.imread('./SfM/Data/sfm04.jpg')

# create SIFT instance
sift = cv.xfeatures2d.SIFT_create()

# detect and compute keypoints
img1_kp, img1_des = sift.detectAndCompute(img1, None)
img2_kp, img2_des = sift.detectAndCompute(img2, None)

img1_drawKps = cv.drawKeypoints(img1, img1_kp, None)
img2_drawKps = cv.drawKeypoints(img2, img2_kp, None)

# save result as image
cv.imwrite('sift_keypoints.jpg',img1_drawKps)

# Brute force matching with k=2
bf = cv.BFMatcher()
matches = bf.knnMatch(img1_des, img2_des, k=2)

# Ratio test and retrieval of indices
good = [m1 for m1, m2 in matches if m1.distance < 0.75*m2.distance]
sorted_matches = sorted(good, key=lambda x: x.distance)

# save matching result as image
res = cv.drawMatches(img1, img1_kp, img2, img2_kp, sorted_matches, img2, flags=2) 
cv.imwrite('sift_bfMatcher.jpg',res)

# Step 2: Essential Matrix Estimation
query_idx = [match.queryIdx for match in sorted_matches]
train_idx = [match.trainIdx for match in sorted_matches]

kp1 = np.float32([img1_kp[ind].pt for ind in query_idx])
kp2 = np.float32([img2_kp[ind].pt for ind in train_idx])

ones = np.ones((1, len(kp1)))

q1 = np.append(kp1.T, ones, axis=0)
q2 = np.append(kp2.T, ones, axis=0)

norm_q1 = K_inv@q1
norm_q2 = K_inv@q2

# 5-points algorithm / RANSAC
best_in = 0
best_E = None

for _ in range(MAXITER):
    idx = np.random.randint(0, len(kp1), size=5)
    rand_norm_q1 = norm_q1[:, idx]
    rand_norm_q1_in = matlab.double(rand_norm_q1.tolist())
    rand_norm_q2 = norm_q2[:, idx]
    rand_norm_q2_in = matlab.double(rand_norm_q2.tolist())
    
    E = eng.calibrated_fivepoint(rand_norm_q1_in, rand_norm_q2_in)
    E = np.array(E)
    
    for i in range(E.shape[1]):
        cur_E = E[:,i].reshape(3,3)
        det = np.linalg.det(cur_E)
        
        const = 2 * cur_E @ cur_E.T @ cur_E - np.trace(cur_E@cur_E.T)*cur_E
        estim = np.diag(norm_q2.T @ cur_E @ norm_q1)
        cur_in = sum(np.where(((estim < threshold) & (estim > 0)), True, False))
        if best_in < cur_in:
            best_in = cur_in
            best_E = cur_E
            inliner = np.where(((estim < threshold) & (estim > 0)))
            
print(f'# of inliers: {best_in}')
inliner = np.array(inliner).reshape(-1)




        

    
    