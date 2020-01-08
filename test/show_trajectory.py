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

    # Show error in degrees
    plt.plot(x_axis, reference_data[:,1], 'b', label='x')
    plt.plot(x_axis, reference_data[:,2], 'r', label='y')
    plt.plot(x_axis, reference_data[:,3], 'g', label='z')
    plt.plot(x_axis, reference_data[:,4], 'c', label='rx')
    plt.plot(x_axis, reference_data[:,5], 'y', label='ry')
    plt.plot(x_axis, reference_data[:,6], 'm', label='rz')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
