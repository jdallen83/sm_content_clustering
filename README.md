# sm_content_clustering
A Python module for clustering creators of social media content into networks.

Currently supports identifying potential networks of Facebook Pages in the CSV output files from CrowdTangle.

## Installation

Can install via pip with

`pip install git+https://github.com/jdallen83/sm_content_clustering`

Install requires [pandas](https://pypi.org/project/pandas/) and [fasttext](https://pypi.org/project/fasttext/)

### Language Prediction

To enable language prediction, you will need to download a [fasttext language model](https://fasttext.cc/docs/en/language-identification.html). Module was tested with lid.176.ftz.

## Usage

### Command line
Can be called as a module for command line usage.

For usage guide:

`python -m sm_content_clustering -h`

Example that will create an output CSV with potential networks and predicted languages from several input CSVs:

`python -m sm_content_clustering --add_language --ft_model_path /path/to/lid.176.ftz --output_path /path/to/output.csv --min_threshold 0.03 /path/to/input_1.csv /path/to/input_2.csv`

### Python

Module can also be called from within Python.

Example that will generate a Pandas dataframe that contains potential networks:

    import sm_content_clustering.sm_processor as sm_processor

    input_files = ['/path/to/1.csv', '/path/to/2.csv', '/path/to/3.csv']
    df = sm_processor.ct_generate_page_clusters(input_files, add_language=True, ft_model_path='/path/to/lid.176.ftz')
    print(df)

### Options

Arguments for sm_processor.ct_generate_page_clusters() are

1. infiles: Input files to read content from. Required.
2. content_cols: Which columns from the input files to use as content for the purposes of clustering identical posts. Default: Message, Image Text, Link, Link Text
3. add_language: Whether to predict the page and network languages. Default: False
4. ft_model_path: Path to fasttext model file. Default: None
5. outfile: Path to write output CSV with potential networks. Default: None
6. update_every: How often to output clustering status. (Print status 1 every N pages). Default: 1000
7. min_threshold: Minimum similarity score for clustering. Possible range between 0 and 1, with 1 being perfect high confidence overlap, and 0 being no overlap. Default: 0.03
8. second_cluster_factor: Requirement that the best matched cluster for a page must score a factor X above the second best matched cluster. Default: 2.5

## Methodology

Module assumes you have social media content, which includes the body content of a message and the account that created it. It begins by grouping by all messages, and finds which accounts have shared identical messages within the dataset. It then applies a basic agglomerative clustering algorithm to group the accounts into clusters that are frequently sharing the same messages.

The clustering loops through the list of all accounts, normally sorted in reverse size or popularity, and for each account, searches all existing clusters to see if there is a valid match, given the min_threshold and second_cluster_factor parameters. If there is a match, the account is added to the existing cluster. If there is not a match, then, if there is enough messages from the account to justify, a new cluster will be created with the account acting as the seed.

## License

MIT License
