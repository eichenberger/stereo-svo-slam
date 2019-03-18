class KeyFrame:
    def __init__(self, image, keypoints2d, keypoints3d, pose):
        self.image = image
        self.keypoints2d = keypoints2d
        self.keypoints3d = keypoints3d
        self.pose = pose
