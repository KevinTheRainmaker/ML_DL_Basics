import numpy as np
import cv2 as cv

import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath("./PA2_Structure_from_Motion/SfM/Step2")  # 'calibrated_fivepoint.m'가 위치한 경로

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

MAXITER = 100
threshold = 5e-4

# intrinsic parameter
K = np.array([[3451.5, 0.0, 2312.0], [0.0, 3451.5, 1734], [0.0,0.0,1.0]])
K_inv = np.linalg.inv(K)

################################################################################################################################
# Step 1: Feature Extraction & Matching
################################################################################################################################
img1 = cv.imread('./PA2_Structure_from_Motion/SfM/Data/sfm03.jpg')
img2 = cv.imread('./PA2_Structure_from_Motion/SfM/Data/sfm04.jpg')

# create SIFT instance
sift = cv.SIFT_create()

# detect and compute keypoints
img1_kp, img1_des = sift.detectAndCompute(img1, None)
img2_kp, img2_des = sift.detectAndCompute(img2, None)

# KNN matching with k=2
bf = cv.BFMatcher()
matches = bf.knnMatch(img1_des, img2_des, k=2)

# Ratio test
good = [m1 for m1, m2 in matches if m1.distance < 0.8 * m2.distance]
sorted_good = sorted(good, key=lambda x: x.distance)

# option: default Brute-force Matching
# matches = bf.match(img1_des,img2_des)
# sorted_good = sorted(matches, key = lambda x : x.distance)

print(f'# of matches: {len(sorted_good)}')

# save result as image
img1_drawKps = cv.drawKeypoints(img1, img1_kp, None)
img2_drawKps = cv.drawKeypoints(img2, img2_kp, None)
cv.imwrite('./PA2_Structure_from_Motion/results/sift_keypoints.jpg',img1_drawKps)
print('save keypoints: done')

res = cv.drawMatches(img1, img1_kp, img2, img2_kp, sorted_good, img2, flags=2) 
cv.imwrite('./PA2_Structure_from_Motion/results/sift_bfMatcher.jpg',res)
print('save matches: done')

################################################################################################################################
# Step 2: Essential Matrix Estimation
################################################################################################################################
query_idx = [good.queryIdx for good in sorted_good]
train_idx = [good.trainIdx for good in sorted_good]

kp1 = np.float32([img1_kp[idx].pt for idx in query_idx])
kp2 = np.float32([img2_kp[idx].pt for idx in train_idx])

ones = np.ones((1, len(kp1)))

q1 = np.append(kp1.T, ones, axis=0)
q2 = np.append(kp2.T, ones, axis=0)

norm_q1 = K_inv @ q1
norm_q2 = K_inv @ q2

best_in = 0
best_E = None

# 5-points algorithm / RANSAC
for _ in tqdm(range(MAXITER)):

    idx = np.random.randint(0, len(kp1), size=5)
    rand_norm_q1 = norm_q1[:, idx]
    rand_norm_q2 = norm_q2[:, idx]
    rand_norm_q1_doub = matlab.double(rand_norm_q1.tolist())
    rand_norm_q2_doub = matlab.double(rand_norm_q2.tolist())
    
    E = eng.calibrated_fivepoint(rand_norm_q1_doub, rand_norm_q2_doub)
    E = np.array(E)
    
    for i in range(E.shape[1]):
        cur_E = E[:,i].reshape(3,3)
        
        estim = np.diag(norm_q2.T @ cur_E @ norm_q1)
        cur_in = sum(np.where(((estim < threshold) & (estim >= 0)), True, False))
        if best_in < cur_in:
            best_in = cur_in
            best_E = cur_E
            inlier_idx = np.where(((estim < threshold) & (estim >= 0)))

      
print(f'# of inliers: {best_in}')
inlier_idx = np.array(inlier_idx).reshape(-1)

# save
df = pd.DataFrame(best_E)
df.to_csv('./PA2_Structure_from_Motion/results/EssentialMatrix.csv')
print('save Essential Matrix: done')

################################################################################################################################
# Step 3: Essential Matrix Decomposition & Step 4: Triangulation
################################################################################################################################
U, S, VT = np.linalg.svd(best_E, full_matrices=True)

W = np.array([[0.0, -1.0, 0.0],
              [1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0]
              ])

P = np.array([
    np.column_stack((U @ W @ VT, U[:,2])),
    np.column_stack((U @ W @ VT, -U[:,2])),
    np.column_stack((U @ W.T @ VT, U[:,2])),
    np.column_stack((U @ W.T @ VT, -U[:,2]))])


print('candidate for camera matrix\n', P)

E_init = np.append(np.eye(3), np.zeros((3,1)), axis=1)
for i in range(4):
    tmp = P[i]
    for j in range(len(inlier_idx)):
        A = np.array([
            norm_q1[0,j+1]*E_init[2] - E_init[0].T,
            norm_q1[1,j+1]*E_init[2] - E_init[1].T,
            norm_q2[0,j+1]*tmp[2] - tmp[0].T,
            norm_q2[1,j+1]*tmp[2] - tmp[1].T])
        U_A, S_A, VT_A = np.linalg.svd(A, full_matrices=True)
        X = VT_A[3]/VT_A[3,3]

        if X[2] > 0 and (tmp@X.T)[2]>0:
            print(f'Matrix {i} have all positive depths!')
            E = P[i]
            break
        else:
            print(f'Matrix {i} is not good')
            break

inlier_X = []
eig = np.linalg.eig
eig_arr = [0,0,0,0]

for i in range(len(inlier_idx)):
    A = np.array([
        norm_q1[0,i+1]*E_init[2].T - E_init[0].T,
        norm_q1[1,i+1]*E_init[2].T - E_init[1].T,
        norm_q2[0,i+1]*E[2].T - E[0].T,
        norm_q2[1,i+1]*E[2].T - E[1].T])
    U_A, S_A, VT_A = np.linalg.svd(A, full_matrices=True)
    eig_arr = eig(VT_A)[0]
    X = VT_A[:,np.argmax(eig_arr)]

    inlier_X.append(X)
    
p3ds = np.array(inlier_X).T

# visualize
X = np.array([])
Y = np.array([])
Z = np.array([])
X = np.concatenate((X, p3ds[0]))
Y = np.concatenate((Y, p3ds[1]))
Z = np.concatenate((Z, p3ds[2]))

fig = plt.figure(figsize=(30,30))
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, c='b', marker='o') 
plt.savefig('./PA2_Structure_from_Motion/results/3D_result.jpg')
plt.show()