import argparse
from spextractor import __version__
from multiprocessing import cpu_count
from snakemake import snakemake
from pkg_resources import resource_filename


def main():
    parser = argparse.ArgumentParser(description='spectrum extractor')
    parser.add_argument('-v', '--version', action='version', version=__version__, help='print version and exit')
    parser.add_argument('--dryrun', action='store_true', help='perform a dry run')
    parser.add_argument('--unlock', action='store_true', help='unlock directory')
    parser.add_argument('--touch', action='store_true', help='touch output files only')
    parser.add_argument('--latency', metavar='N', type=int, default=3, help='specify filesystem latency (seconds)')
    parser.add_argument('--cores', metavar='N', type=int, default=cpu_count(),
                        help='number of cores used for execution (local execution only)')
    parser.add_argument('--count', metavar='N', type=int,
                        help='number of files to process (limits DAG size)')
    parser.add_argument('--start', metavar='IDX', type=int, default=0,
                        help='starting file index (for use with --count)')

    # parse args
    args = parser.parse_args()

    # start/stop config
    if args.count is not None:
        config = {'start': args.start, 'stop': args.start + args.count}
    else:
        config = {}

    snakemake(resource_filename('spextractor', 'Snakefile'),
              config=config,
              keepgoing=True,
              force_incomplete=True,
              cores=args.cores,
              dryrun=args.dryrun,
              unlock=args.unlock,
              touch=args.touch,
              latency_wait=args.latency)


if __name__ == '__main__':
    main()
