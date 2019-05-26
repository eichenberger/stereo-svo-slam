class KeyFrame:
    def __init__(self, left, right, keypoints2d, keypoints3d, pose, colors):
        self.left = left
        self.right = right
        self.keypoints2d = keypoints2d
        self.keypoints3d = keypoints3d
        self.pose = pose
        self.colors = colors
