class DepthAdjustment:
    def adjust_depth(keyframe, pose):
        keyframe_pose = keyframe.pose
        keypoints3d = keyframe.keypoints3d
        keypoints2d = pose * keypoints3d

        corrected_rot = np.linalg.pinv(keyframe_pose[:,0:3])

        keypoints3d_corrected = keypoints3d - keyframe_pose[:,3]
        keypoints3d_corrected = corrected_rot * keypoints3d_corrected

        pose_corrected = pose
        pose_corrected[:,3] = pose[:,3] - keyframe_pose[:,3]
        pose_corrected[:,0:3] = np.linalg.pinv(pose_corrected[:,0:3])

