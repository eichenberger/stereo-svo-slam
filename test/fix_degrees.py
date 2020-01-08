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
    parser.add_argument('-o', '--out', help='out csv', type=str)

    args = parser.parse_args()
    reference = args.ref
    out = args.out

    reference_data = np.array(read_csv(reference), dtype='f')
    reference_data[:,4:] = reference_data[:,4:]/180*math.pi;


    with open(out, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in reference_data:
            spamwriter.writerow(row)

if __name__ == "__main__":
    main()
