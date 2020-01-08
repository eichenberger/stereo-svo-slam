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
    parser.add_argument('-t', '--test', help='test csv', type=str)

    args = parser.parse_args()
    reference = args.ref
    test = args.test

    test_data = np.array(read_csv(test), dtype='f')
    reference_data = np.array(read_csv(reference), dtype='f')[0:test_data.shape[0],:]

    x_axis = list(range(0, test_data.shape[0]))

    diff = test_data -reference_data
    # Show error in degrees
    diff[:,4:] = diff[:,4:]/math.pi*180;
    plt.plot(x_axis, diff[:,1], 'b', label='x')
    plt.plot(x_axis, diff[:,2], 'r', label='y')
    plt.plot(x_axis, diff[:,3], 'g', label='z')
    plt.plot(x_axis, diff[:,4], 'c', label='rx')
    plt.plot(x_axis, diff[:,5], 'y', label='ry')
    plt.plot(x_axis, diff[:,6], 'm', label='rz')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
