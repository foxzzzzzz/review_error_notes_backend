"""Integer neighborhood counts equivalent to the reference cross scorer."""

import cv2
import numpy as np


def neighborhood_mean(mask, kernel):
    # filter2D may use FFT internally. Round binary sums back to exact counts.
    counts = np.rint(cv2.filter2D(mask.astype(np.float64), -1, kernel,
                                 borderType=cv2.BORDER_CONSTANT))
    valid = np.rint(cv2.filter2D(np.ones(mask.shape, dtype=np.float64), -1, kernel,
                                borderType=cv2.BORDER_CONSTANT))
    return np.divide(counts, valid, out=np.zeros_like(counts), where=valid > 0)


def score_cross_centers(red_mask, geometry_mask, arm_offsets, center_radius,
                        center_min_density, arm_min_density):
    center_kernel = np.ones((2 * center_radius + 1,) * 2, dtype=np.float64)
    center_density = neighborhood_mean(red_mask, center_kernel)
    arm_densities = []
    for offsets in arm_offsets.values():
        radius = int(np.max(np.abs(offsets)))
        kernel = np.zeros((2 * radius + 1,) * 2, dtype=np.float64)
        kernel[offsets[:, 0] + radius, offsets[:, 1] + radius] = 1
        arm_densities.append(neighborhood_mean(geometry_mask, kernel))
    minimum = np.minimum.reduce(arm_densities)
    accepted = red_mask & (center_density >= center_min_density) & (minimum >= arm_min_density)
    return [
        {"x": int(x), "y": int(y), "min_arm_density": float(minimum[y, x]),
         "arm_densities": [float(density[y, x]) for density in arm_densities],
         "center_density": float(center_density[y, x])}
        for y, x in np.argwhere(accepted)
    ]
