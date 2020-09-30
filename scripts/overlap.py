import argparse
from collections import defaultdict  # available in Python 2.5 and newer


def read_ud(filename, column, exclude=None):
    token_count = defaultdict(int)
    total_count = 0
    for line in open(filename):
        line = line.rstrip()
        if line.startswith('#') or len(line) == 0:
            continue
        else:
            token = line.split('\t')[column]
            if exclude is None or token not in exclude:
                token_count[token] += 1
                total_count += 1

    return token_count, total_count


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-S", "--src", type=str)
    parser.add_argument("-T", "--tgt", type=str)
    parser.add_argument("-C", "--column", type=int, default=1)
    parser.add_argument('-E', '--exclude', nargs='*', type=str)

    args = parser.parse_args()

    sc, st = read_ud(args.src, args.column, args.exclude)
    print('Source file: {}\nType/Token count: {} / {}'.format(args.src, len(sc), st))

    tc, tt = read_ud(args.tgt, args.column, args.exclude)
    print('Target file: {}\nType/Token count: {} / {}'.format(args.tgt, len(tc), tt))

    total_overlap = 0
    unique_overlap = 0
    for word, count in tc.items():
        if word in sc:
            total_overlap += tc[word]
            unique_overlap += 1
    total_ratio = total_overlap / tt * 100
    unique_ratio = unique_overlap / len(tc) * 100
    print('Type/Token overlap ratio: {:.2f} / {:.2f}'.format(unique_ratio, total_ratio))


if __name__ == "__main__":
    main()
