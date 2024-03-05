#!/usr/bin/python3
import argparse
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', dest='filepath', help='Filepath that will be displayed')
    parser.add_argument('--begin', default='0', dest='begin', help='Begin timestamp')
    parser.add_argument('--end', default='-1', dest='end', help='End timestamp')

    args = parser.parse_args()

    return args


def load_file(filepath: str, begin: float, end: float) -> list[(float, float)]:
    file = open(filepath, 'r')
    lines = file.readlines()
    
    values: list[float] = []
    for line in lines:
        tempValues = line.split()
        timestamp = float(tempValues[0])

        if begin <= timestamp and (timestamp <= end or end == -1):
            values.append((timestamp, float(tempValues[1])))

    return values


def show_plot(values: list[(float, float)]) -> None:
    x_values = [value[0] for value in values]
    y_values = [value[1] for value in values]

    plt.plot(x_values, y_values)
    plt.xlabel('Czas')
    plt.ylabel('Wartość EKG')
    plt.title('Wykres EKG w czasie')
    plt.grid(True)
    plt.show()


'''
    Example of running this script:
    view.py --filepath ekg1_converted.txt --begin 0 --end -1
'''
if __name__ == '__main__':
    args = get_args()
    values = load_file(args.filepath, int(args.begin), int(args.end))
    show_plot(values)
