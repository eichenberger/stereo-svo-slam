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
    parser.add_argument('-a', '--test1', help='test csv', type=str)
    parser.add_argument('-b', '--test2', help='test csv', type=str)
    parser.add_argument('-c', '--test3', help='test csv', type=str)
    parser.add_argument('-d', '--test4', help='test csv', type=str)

    args = parser.parse_args()

    test_data1 = np.array(read_csv(args.test1), dtype='f')
    test_data2 = np.array(read_csv(args.test2), dtype='f')
    if args.test3:
        test_data3 = np.array(read_csv(args.test3), dtype='f')
    if args.test4:
        test_data4 = np.array(read_csv(args.test4), dtype='f')
    reference_data = np.array(read_csv(args.ref), dtype='f')

    xaxis = 3
    yaxis = 2

    x_axis = reference_data[:,xaxis]
    y_axis = reference_data[:,yaxis]
    plt.plot(x_axis, y_axis, 'r', label='Ground Truth')
    x_axis = test_data1[:,xaxis]
    y_axis = test_data1[:,yaxis]
    plt.plot(x_axis, y_axis, 'g', label='Our SVO')
    x_axis = test_data2[:,xaxis]
    y_axis = test_data2[:,yaxis]
    plt.plot(x_axis, y_axis, 'b', label='ORB SLAM')
    if args.test3:
        x_axis = test_data3[:,xaxis]
        y_axis = test_data3[:,yaxis]
        plt.plot(x_axis, y_axis, 'y', label='SVO')
    if args.test4:
        x_axis = test_data4[:,xaxis]
        y_axis = test_data4[:,yaxis]
        plt.plot(x_axis, y_axis, 'c', label='SVO Edglet')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
