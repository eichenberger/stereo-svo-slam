import matplotlib.pyplot as plt
import csv
import argparse
import numpy as np
import math

def read_csv(file):
    _rows = []
    with open(file,'r') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        for row in rows:
            _rows.append(row)

    return _rows

def main():
    parser = argparse.ArgumentParser(description='evaluate results')

    parser.add_argument('-r', '--ref', help='euroc reference csv', type=str)
    parser.add_argument('-t', '--test', help='test data csv', type=str)
    parser.add_argument('-s', '--skip', help='skip n ref frames', type=str)
    parser.add_argument('-m', '--move', help='skip n test frames', type=str)
    parser.add_argument('-a', '--angle', help='correction angle', type=str)

    args = parser.parse_args()

    # Skip first line
    reference_data = np.array(read_csv(args.ref), dtype='f')
    test_data = np.array(read_csv(args.test), dtype='f')

    # Make sure we start at 0,0,0
    ref_pose_data = reference_data[int(args.skip):,:]
    ref_pose_data = ref_pose_data - ref_pose_data[0]

    test_pose_data = test_data[int(args.move):,1:4]
    test_pose_data = test_pose_data -test_pose_data[0]

    axis_test = np.array([test_pose_data[:,1],
                          test_pose_data[:,2]])

    angle = float(args.angle)/180.0*math.pi
    R=np.mat([[math.cos(angle), -math.sin(angle)],
              [math.sin(angle), math.cos(angle)]])

    axis_test = np.matmul(R,axis_test)

    x_axis_ref = -ref_pose_data[:,0]
    y_axis_ref = -ref_pose_data[:,2]

    # Show error in degrees
    x_axis_test = np.asarray(axis_test[0])[0]
    y_axis_test = -np.asarray(axis_test[1])[0]
    plt.plot(x_axis_ref, y_axis_ref, 'r', label='Ground Truth')
    plt.plot(x_axis_test, y_axis_test, 'b', label='Our SVO')
    plt.xlabel("z (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
