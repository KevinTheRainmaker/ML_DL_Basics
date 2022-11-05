import numpy as np
import cv2 as cv

import glob

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

# create SIFT instance
sift = cv.SIFT_create()

ims = []
kps = []
des = []

images = glob.glob('./PA2_Structure_from_Motion/SfM/Data/*.jpg')
for fname in tqdm(images):
    img = cv.imread(fname)

    # detect and compute keypoints
    img_kp, img_des = sift.detectAndCompute(img, None)
    ims.append(img)
    kps.append(img_kp)
    des.append(img_des)

matches_save = np.array([[None]*len(images)]*len(images))
print(matches_save.shape)
for i in tqdm(range(len(images))):
    for j in range(len(images)):
        if i != j:
            # KNN matching with k=2
            bf = cv.BFMatcher()
            matches = bf.knnMatch(des[i], des[j], k=2)
            # Ratio test
            good = [m1 for m1, m2 in matches if m1.distance < 0.8 * m2.distance]
            sorted_good = sorted(good, key=lambda x: x.distance)
            matches_save[i,j] = sorted_good

print('matching done')

################################################################################################################################
# Step 2: Essential Matrix Estimation
################################################################################################################################
query_idx = [good.queryIdx for good in matches_save[0,1]]
train_idx = [good.trainIdx for good in matches_save[0,1]]

kp1 = np.float32([kps[0][idx].pt for idx in query_idx])
kp2 = np.float32([kps[1][idx].pt for idx in train_idx])

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

################################################################################################################################
# Step 5: Growing Step
################################################################################################################################
for i in range(1,5):
    query_idx = [good.queryIdx for good in matches_save[i,i+1]]
    train_idx = [good.trainIdx for good in matches_save[i,i+1]]

    kp1 = np.float32([kps[i][idx].pt for idx in query_idx])
    kp2 = np.float32([kps[i+1][idx].pt for idx in train_idx])

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
plt.savefig('./PA2_Structure_from_Motion/results/3D_result_multi.jpg')
plt.show()