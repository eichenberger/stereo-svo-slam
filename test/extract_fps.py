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

    parser.add_argument('-t', '--test', help='test csv', type=str)

    args = parser.parse_args()
    test = args.test

    test_data = np.array(read_csv(test), dtype='f')

    dt = 0
    i = 1
    test_point_prev = test_data[0]
    for test_point in test_data[1:]:
        dt += test_point[0]-test_point_prev[0]
        i += 1
        test_point_prev = test_point

    print(f"Average FPS: {1/(dt/i):.2f}")




if __name__ == "__main__":
    main()
