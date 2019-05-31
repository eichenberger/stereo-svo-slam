class KeyFrame:
    def __init__(self, left, right, keypoints2d, keypoints3d, pose, colors):
        self.left = left
        self.right = right
        self.keypoints2d = keypoints2d
        self.keypoints3d = keypoints3d
        self.pose = pose
        self.colors = colors

        self.blacklist = [None]*len(left)
        for i in range(0, len(self.blacklist)):
            self.blacklist[i] = [False]*self.keypoints2d[i].shape[1]

