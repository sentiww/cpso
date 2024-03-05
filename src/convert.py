#!/usr/bin/python3
import argparse
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', dest='filepath', help='Filepath to convert')
    parser.add_argument('--type', choices=['raw-single-column', 'raw-multi-column', 'timed'], dest='type', help='Type of format the file is saved in')
    parser.add_argument('--sampling', default='100', dest='sampling', help='Sampling (Hz) that was used to create the file')

    args = parser.parse_args()

    return args


def load_file(filepath: str) -> list[str]:
    file = open(filepath, 'r')
    lines = file.readlines()

    return lines


def process_raw_single_column(lines: list[str], sampling: int) -> list[(float, float)]:
    offset: float = 1 / sampling
    timestamp: float = 0
    entries: list[(float, float)] = []
    for line in lines:
        entries.append((timestamp, float(line)))
        timestamp = timestamp + offset

    return entries


def process_raw_multi_column(lines: list[str], sampling: int) -> list[(float, float)]:
    offset: float = 1 / sampling
    timestamp: float = 0
    entries: list[(float, float)] = []
    for line in lines:
        values = line.split()
        for value in values:
            entries.append((timestamp, float(value)))
            timestamp = timestamp + offset

    return entries


def process_timed(lines: list[str], sampling: int) -> list[(float, float)]:
    entries: list[(float, float)] = []
    for line in lines:
        values = line.split()
        entries.append((float(values[0]), float(values[1])))

    return entries


def get_output_filepath(filepath: str) -> str:    
    directory_split = filepath.split('/')[:-1]
    output_directory = '/'.join(directory_split)

    filepath_split = filepath.split('/')[-1].split('.')
    filename = filepath_split[0]
    extension = '.'.join(filepath_split[1:]) 

    return f'{output_directory}/{filename}_converted.{extension}'


def save_file(filepath: str, entries: list[(float, float)]) -> None:
    file = open(filepath, 'w')    

    for entry in entries:
        file.write(f'{entry[0]:.8E}\t{entry[1]:.8E}\n')

    file.close()


'''
    Example of running this script:
    convert.py --filepath ekg_noise.txt --sampling 360 --type timed
'''
if __name__ == '__main__':
    args = get_args()
    lines = load_file(args.filepath)

    entries: list[(int, int)] = []
    if args.type == 'raw-single-column':
        entries = process_raw_single_column(lines, int(args.sampling))
    elif args.type == 'raw-multi-column':
        entries = process_raw_multi_column(lines, int(args.sampling))
    elif args.type == 'timed':
        entries = process_timed(lines, int(args.sampling))

    output_filepath = get_output_filepath(args.filepath)
    save_file(output_filepath, entries)
