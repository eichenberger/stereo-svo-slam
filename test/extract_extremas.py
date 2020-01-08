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

    adiff = np.absolute(diff)
    maximas = np.amax(adiff, axis=0)

    print("Maximas:")
    print(f"{maximas[1]:.3f} & {maximas[2]:.3f} & {maximas[3]:.3f} & {maximas[4]:.3f} & {maximas[5]:.3f} & {maximas[6]:.3f}")
    print(f"duration: {maximas[0]}")

    average = np.average(adiff, axis=0)
    print(f"{average[1]:.3f} & {average[2]:.3f} & {average[3]:.3f} & {average[4]:.3f} & {average[5]:.3f} & {average[6]:.3f}")

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
