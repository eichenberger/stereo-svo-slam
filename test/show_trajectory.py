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

    parser.add_argument('-r', '--ref', help='reference csv', type=str)

    args = parser.parse_args()

    reference_data = np.array(read_csv(args.ref), dtype='f')

    x_axis = list(range(0, reference_data.shape[0]))

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.set_xlabel('Frame number')
    ax1.set_ylabel('Position (m)')
    # Show error in degrees
    ax1.plot(x_axis, reference_data[:,1], 'b', label='x')
    ax1.plot(x_axis, reference_data[:,2], 'r', label='y')
    ax1.plot(x_axis, reference_data[:,3], 'g', label='z')
    ax2.set_xlabel('Frame number')
    ax2.set_ylabel('Angle (°)')
    ax2.plot(x_axis, reference_data[:,4]/math.pi*180.0, 'c', label='rx')
    ax2.plot(x_axis, reference_data[:,5]/math.pi*180.0, 'y', label='ry')
    ax2.plot(x_axis, reference_data[:,6]/math.pi*180.0, 'm', label='rz')

    fig.tight_layout()
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
