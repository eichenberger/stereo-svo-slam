import cv2

def mat_to_points(matrix):
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    points = [None]*matrix.shape[0]
    for i in range(0, matrix.shape[0]):
        points[i] = cv2.KeyPoint(matrix[i, 0], matrix[i, 1], 1)

    return points
