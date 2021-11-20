if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='sm_content_clustering', description='Clusters content from social media sources into networks of accounts that are sharing identical content.')
    parser.add_argument('input_files', metavar='INPUT_FILE', type=str, nargs='+',
                        help='Input data files.')
    parser.add_argument('--add_language', dest='add_language', action='store_const',
                        const=True, default=False,
                        help='Add predicted cluster language')
    parser.add_argument('--ft_model_path', dest='ft_model_path', type=str,
                        help='Path to the FastText model file. Recommended: lid.176.ftz')
    parser.add_argument('--output_path', dest='output_path', type=str,
                        help='Destination path to write the clusters file.')
    parser.add_argument('--min_threshold', dest='min_threshold', type=float,
                        help='Minimum required clustering threshold (0.03)')
    parser.add_argument('--second_cluster_factor', dest='second_cluster_factor', type=float,
                        help='Second cluster factor. The required minimum ratio between best scoring cluster and second best scoring cluster (2.5)')
    parser.add_argument('--update_every', dest='update_every', type=int,
                        help='How often to print updates to the clustering process (1000)')


    args = parser.parse_args()

    kwargs = {}
    if args.min_threshold:
        kwargs['min_threshold'] = args.min_threshold
    if args.second_cluster_factor:
        kwargs['second_cluster_factor'] = args.second_cluster_factor
    if args.update_every:
        kwargs['update_every'] = args.update_every
    if args.add_language:
        kwargs['add_language'] = args.add_language
    if args.ft_model_path:
        kwargs['ft_model_path'] = args.ft_model_path
    if args.output_path:
        kwargs['outfile'] = args.output_path

    from . import sm_processor
    sm_processor.ct_generate_page_clusters(args.input_files, **kwargs)
