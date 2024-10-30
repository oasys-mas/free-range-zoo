"""Examine the stats of a file."""
import sys

sys.path.append('.')

import argparse
from pstats import Stats


def main() -> None:
    """Read a file and print the stats."""
    args = parse_args()

    stats = Stats(args.file).strip_dirs().sort_stats(*args.sort_by)
    stats.print_stats(args.amount, args.string_filter)


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='The file to examine')
    parser.add_argument('--sort_by',
                        type=str,
                        nargs='*',
                        choices=['calls', 'cumtime', 'file', 'ncalls', 'pcalls', 'line', 'name', 'nfl', 'stdname', 'tottime'],
                        default=['tottime'],
                        help='The way to sort the results')
    parser.add_argument('--amount', type=float, default=0.2, help='The amount of data to display')
    parser.add_argument('--string_filter', type=str, default=None, help='Filter the results by a string')
    return parser.parse_args()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('file', type=str, help='The file to examine')
    parser.add_argument('--sort_by',
                        type=str,
                        nargs='*',
                        choices=['calls', 'cumtime', 'file', 'ncalls', 'pcalls', 'line', 'name', 'nfl', 'stdname', 'tottime'],
                        default=['tottime'],
                        help='The way to sort the results')

    parser.add_argument('--amount', type=float, default=0.2, help='The amount of data to display')
    parser.add_argument('--string_filter', type=str, default=None, help='Filter the results by a string')

    args = parser.parse_args()

    main(args)
