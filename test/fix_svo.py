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
            row = list(map(lambda x: float(x), row));
            _rows.append(row)

    return _rows

def main():
    parser = argparse.ArgumentParser(description='evaluate results')

    parser.add_argument('-r', '--ref', help='reference csv', type=str)
    parser.add_argument('-s', '--svo', help='svo csv', type=str)
    parser.add_argument('-o', '--out', help='out csv', type=str)

    args = parser.parse_args()

    reference_data = read_csv(args.ref)
    svo_data = read_csv(args.svo)

    initialized = False
    offsets = [0,0,0,0,0,0]
    correction = [0,0,0,0,0,0]
    # svo edgled
    # scale = [4.771162338583954,2.9593790693740774, 3.2330913889945583, 1, -1, 1]

    # svo rpg
    scale = [3.681885125184094,2.6856019508212547, 3.64182692307692, 1, -1, 1]
    for i in range(0, len(svo_data)):
        if math.fabs(svo_data[i][1]) > 0.01 and not initialized:
            for j in range(1, len(svo_data[i])):
                offsets[j-1] = reference_data[i][j]
                correction[j-1] = svo_data[i][j]
                initialized = True

        for j in range(1, 7):
            svo_data[i][j] -= correction[j-1]
            svo_data[i][j] *= scale[j-1]
            svo_data[i][j] += offsets[j-1]

    with open(args.out, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in svo_data:
            spamwriter.writerow(row)

if __name__ == "__main__":
    main()
