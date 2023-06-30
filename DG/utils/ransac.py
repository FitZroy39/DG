import numpy as np
from skimage.measure import LineModelND, ransac

def get_inliers(X):
    data = X.T
    # fit line using all data
    model = LineModelND()
    model.estimate(data)
    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                                   residual_threshold=5, max_trials=50)
    return inliers

def denoise(lane_out):
    lane_out_ransac = np.zeros_like(lane_out)
    for idx in range(9):
        a = lane_out.clone()
        a[a!=idx]=0
        X = np.array(np.where(a==idx))
        if X.shape[1] > 2:
            inliers = get_inliers(X)
            lane_out_ransac[X[:,inliers][0],X[:,inliers][1]] = idx
    return lane_out_ransac