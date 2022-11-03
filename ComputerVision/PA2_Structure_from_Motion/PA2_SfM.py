import numpy as np
import cv2 as cv

import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath("/Users/kangbeenko/Desktop/GitHub_Repository/ML_DL_Basics/ComputerVision/PA2_Structure_from_Motion/SfM/Step2")  # 'calibrated_fivepoint.m'가 위치한 경로

import pandas as pd
import matplotlib.pyplot as plt

# MAXITER = 100
MAXITER = 1
threshold = 5e-4

# intrinsic parameter
K = np.array([[3451.5, 0.0, 2312.0], [0.0, 3451.5, 1734], [0.0,0.0,1.0]])
K_inv = np.linalg.inv(K)

# Step 1: Feature Extraction & Matching
img1 = cv.imread('/Users/kangbeenko/Desktop/GitHub_Repository/ML_DL_Basics/ComputerVision/PA2_Structure_from_Motion/SfM/Data/sfm03.jpg')
img2 = cv.imread('/Users/kangbeenko/Desktop/GitHub_Repository/ML_DL_Basics/ComputerVision/PA2_Structure_from_Motion/SfM/Data/sfm04.jpg')

# create SIFT instance
sift = cv.xfeatures2d.SIFT_create()

# detect and compute keypoints
img1_kp, img1_des = sift.detectAndCompute(img1, None)
img2_kp, img2_des = sift.detectAndCompute(img2, None)

img1_drawKps = cv.drawKeypoints(img1, img1_kp, None)
img2_drawKps = cv.drawKeypoints(img2, img2_kp, None)

# save result as image
cv.imwrite('PA2_Structure_from_Motion/results/sift_keypoints.jpg',img1_drawKps)

# Brute force matching with k=2
bf = cv.BFMatcher()
matches = bf.knnMatch(img1_des, img2_des, k=2)

# Ratio test and retrieval of indices
good = [m1 for m1, m2 in matches if m1.distance < 0.75*m2.distance]
sorted_matches = sorted(good, key=lambda x: x.distance)

# save matching result as image
res = cv.drawMatches(img1, img1_kp, img2, img2_kp, sorted_matches, img2, flags=2) 
cv.imwrite('PA2_Structure_from_Motion/results/sift_bfMatcher.jpg',res)

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

# save
df = pd.DataFrame(best_E)
df.to_csv('./results/EssentialMatrix.csv')


# Step 3: Essential Matrix Decomposition
U, S, VT = np.linalg.svd(best_E, full_matrices=True)

W = np.array([[0.0, -1.0, 0.0],
              [1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0]
              ])

Z = np.array([[0.0, 1.0, 0.0],
              [-1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0]
              ])

P = np.array([
    np.column_stack((U @ W @ VT, U[:,2])),
    np.column_stack((U @ W @ VT, -U[:,2])),
    np.column_stack((U @ W.T @ VT, U[:,2])),
    np.column_stack((U @ W.T @ VT, -U[:,2]))])

for i in range(4):
    tmp = P[i]
    for j in range(len(kp1)):
        a = kp1[j].flatten()
        b = kp2[j].flatten()
        c = np.concatenate((a, b))
        d = tmp @ c.T
        if np.any(d<0):
            break
        else:
            camera_matrix = P[i]
            
# Step 4: Triangulation
Rt0 = np.hstack((np.eye(3),np.zeros((3, 1))))
Rt1 = K @ camera_matrix

# Triangulation
def LinearTriangulation(Rt0, Rt1, q1, q2):
    A = [q1[1]*Rt0[2,:] - Rt0[1,:],  
        -(q1[0]*Rt0[2,:] - Rt0[0,:]),  
        q2[1]*Rt1[2,:] - Rt1[1,:],  
        -(q2[0]*Rt1[2,:] - Rt1[0,:])]
        
    A = np.array(A).reshape((4,4))
    AA = A.T @ A 
    U_A, S_Aa, VT_A = np.linalg.svd(AA) # right singular vector
 
    return VT_A[3,0:3]/VT_A[3,3]

p3ds = []
for pt1, pt2 in zip(q1, q2):
    p3d = LinearTriangulation(Rt0, Rt1, pt1, pt2)
    p3ds.append(p3d)
p3ds = np.array(p3ds).T

# visualize
X = np.array([])
Y = np.array([])
Z = np.array([])
X = np.concatenate((X, p3ds[0]))
Y = np.concatenate((Y, p3ds[1]))
Z = np.concatenate((Z, p3ds[2]))

fig = plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, c='b', marker='o') 
# plt.show()
plt.savefig('PA2_Structure_from_Motion/results/3D_result.jpg')