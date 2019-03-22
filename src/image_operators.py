import numpy as np

def get_sub_pixel(image, pt):
    pt = np.array(pt)
    pt_floor = np.floor(pt)
    pt_ceil = np.ceil(pt + [2e-10, 2e-10])

    prob_floor = [1,1] - (pt - pt_floor)
    prob_ceil = [1,1] - (pt_ceil - pt)

    sub_pixel_val = prob_floor[0]*prob_floor[1]*image[int(pt_floor[1]), int(pt_floor[0])]
    sub_pixel_val += prob_floor[0]*prob_ceil[1]*image[int(pt_ceil[1]), int(pt_floor[0])]
    sub_pixel_val += prob_ceil[0]*prob_floor[1]*image[int(pt_floor[1]), int(pt_ceil[0])]
    sub_pixel_val += prob_ceil[0]*prob_ceil[1]*image[int(pt_ceil[1]), int(pt_ceil[0])]

    return sub_pixel_val

def get_intensity_diff(image1, image2, keypoint1, keypoint2, errorval=0):
    x1 = int(keypoint1[0])
    y1 = int(keypoint1[1])
    x2 = keypoint2[0]
    y2 = keypoint2[1]

    # If keypoint is outside of second image we ignore it
    if x2 - 2 < 0 or x2 + 2 > image2.shape[1] or \
            y2 - 2 < 0 or y2 + 2 > image2.shape[0]:
        return errorval

    diff = abs(image1[y1, x1] - get_sub_pixel(image2, [x2, y2]))
    diff += abs(image1[y1, x1-1] - get_sub_pixel(image2, [x2-1, y2]))
    diff += abs(image1[y1-1, x1] - get_sub_pixel(image2, [x2, y2-1]))
    diff += abs(image1[y1, x1+1] - get_sub_pixel(image2, [x2+1, y2]))
    diff += abs(image1[y1+1, x1] - get_sub_pixel(image2, [x2, y2+1]))

    return diff


def get_total_intensity_diff(image1, image2, keypoints1, keypoints2):
    diff = np.zeros((keypoints2.shape[1]))
    for i in range(keypoints2.shape[1]):
        diff[i] = get_intensity_diff(image1, image2, keypoints1[:,i], keypoints2[:,i])

    return diff

