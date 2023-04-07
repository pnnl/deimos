import argparse
from multiprocessing import cpu_count

from pkg_resources import resource_filename
from snakemake import snakemake

from deimos import __version__


def main():
    parser = argparse.ArgumentParser(
        description='DEIMoS: Data Extraction for Integrated Multidimensional Spectrometry')
    parser.add_argument('-v', '--version', action='version',
                        version=__version__, help='print version and exit')
    parser.add_argument('--config', metavar='PATH',
                        default='config.yaml', help='path to yaml configuration file')
    parser.add_argument('--dryrun', action='store_true',
                        help='perform a dry run')
    parser.add_argument('--unlock', action='store_true',
                        help='unlock directory')
    parser.add_argument('--touch', action='store_true',
                        help='touch output files only')
    parser.add_argument('--latency', metavar='N', type=int,
                        default=3, help='specify filesystem latency (seconds)')
    parser.add_argument('--cores', metavar='N', type=int, default=cpu_count(),
                        help='number of cores used for execution (local execution only)')
    parser.add_argument('--count', metavar='N', type=int,
                        help='number of files to process (limits DAG size)')
    parser.add_argument('--start', metavar='IDX', type=int, default=0,
                        help='starting file index (for use with --count)')

    # Cluster-specific options
    clust = parser.add_argument_group('cluster arguments')
    clust.add_argument('--cluster', metavar='PATH',
                       help='path to cluster execution yaml configuration file')
    clust.add_argument('--jobs', metavar='N', type=int, default=1000,
                       help='number of simultaneous jobs to submit to a slurm queue')

    # Parse args
    args = parser.parse_args()

    # Start/stop config
    if args.count is not None:
        config = {'start': args.start, 'stop': args.start + args.count}
    else:
        config = {}

    # Cluster config
    if args.cluster is not None:
        cluster = "sbatch -A {cluster.account} -N {cluster.nodes} -t {cluster.time} -J {cluster.name} --ntasks-per-node {cluster.ntasks} -p {cluster.partition}"
    else:
        cluster = None

    # Call snakemake
    snakemake(resource_filename('workflows', 'default.smk'),
              configfiles=[args.config],
              config=config,
              cluster_config=args.cluster,
              cluster=cluster,
              keepgoing=True,
              force_incomplete=True,
              cores=args.cores,
              nodes=args.jobs,
              dryrun=args.dryrun,
              unlock=args.unlock,
              touch=args.touch,
              latency_wait=args.latency)


if __name__ == '__main__':
    main()
