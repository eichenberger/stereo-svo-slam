import bpy
import csv
import numpy as np
from itertools import chain

# Where to store the trajectory
filepath = "/tmp/blender-classroom.csv"

with open(filepath, 'w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file, dialect='excel')

    scene = bpy.context.scene
    frame_current = scene.frame_current

    loc_offset = None
    rot_offset = None
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        for ob in scene.objects:
            # renderCam is the name of the camera
            if "renderCam" in ob.name:
                mat = ob.matrix_world
                loc = np.array(mat.translation.xzy)
                rot = np.array(mat.to_euler())
                if loc_offset is None:
                    loc_offset = loc
                if rot_offset is None:
                    rot_offset = rot

                loc = (loc - loc_offset)*[1,-1,1]
                rot = (rot - rot_offset)*[1,1,-1]
                rot = (rot[0], rot[2], rot[1])

                loc = tuple(loc.tolist())
                csv_writer.writerow(tuple(chain((0,), loc, rot)))

    scene.frame_set(frame_current)
