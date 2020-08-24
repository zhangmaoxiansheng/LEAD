import numpy as np
import cv2
cv2.setNumThreads(0)
class PnP():
    def __init__(self):
        self.K = np.array([[0.58, 0, 0.5],
                           [0, 1.92, 0.5],
                           [0, 0, 1]], dtype=np.float32)

    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3],[0.299, 0.587, 0.114])
    def feature_match(self,img1,img2):
        max_n_features = 500
        use_flann = False

        detector = cv2.xfeatures2d.SIFT_create(max_n_features)
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)
        if (des1 is None) or (des2 is None):
            return [] ,[]
        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)

        if use_flann:
            # FLANN parameters
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
        else:
            matcher = cv2.DescriptorMatcher().create('BruteForce')
            matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        pts1 = []
        pts2 = []

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
        
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        return pts1, pts2
    
    def conv2d_3d(self,u,v,z,K):
        v0 = K[1][2]
        u0 = K[0][2]
        fy = K[1][1]
        fx = K[0][0]
        x = (u - u0) * z / fx
        y = (v - v0) * z / fy
        return (x,y,z)



    def __call__(self, rgb_curr, rgb_near, depth_curr):
        K = self.K.copy()
        width = rgb_curr.shape[0]
        height = rgb_curr.shape[1]
        K[0, :] *= width
        K[1, :] *= height
        gray_curr = self.rgb2gray(rgb_curr).astype(np.uint8)
        gray_near = self.rgb2gray(rgb_curr).astype(np.uint8)
        height, width = gray_curr.shape
        T = np.eye(4,dtype='float32')
        R = np.eye(4,dtype='float32')
        t = np.eye(4,dtype='float32')

        pts2d_curr, pts2d_near = self.feature_match(gray_curr, gray_near)
        
        pts3d_curr = []
        pts2d_near_filtered = []
        for i, pt2d in enumerate(pts2d_curr):
            u,v = pt2d[0], pt2d[1]
            z = depth_curr[v, u]
            if z > 1:
                xyz_curr = self.conv2d_3d(u,v,z,K)
                pts3d_curr.append(xyz_curr)
                pts2d_near_filtered.append(pts2d_near[i])
        
        if len(pts3d_curr) >= 4 and len(pts2d_near_filtered) >= 4:
            pts3d_curr = np.expand_dims(np.array(pts3d_curr).astype(np.float32),axis=1)
            pts2d_near_filtered = np.expand_dims(np.array(pts2d_near_filtered).astype(np.float32),axis=1)
            ret = cv2.solvePnPRansac(pts3d_curr, pts2d_near_filtered, K,distCoeffs=None)
            success = ret[0]
            rotation_vec = ret[1]
            translation_vec = ret[2]
            r_mat,_ = cv2.Rodrigues(rotation_vec)
            R[:3,:3] = r_mat
            t[:3,3] = translation_vec.reshape(3)
            T[:3,:3] = r_mat
            T[:3,3] = translation_vec.reshape(3)
            return success,R,t,T
        else:
            return 0,R,r,T

