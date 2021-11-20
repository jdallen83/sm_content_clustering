"""
This module contains functions, helpers, for reading in files with social media content.
"""

__version__ = '0.1'
__author__ = 'Jeff Allen'


import pandas as pd
import urllib
import math
import fasttext
from collections import Counter

from . import clustering
from . import language_prediction


# These are the fields within CT data files that contain meaningful content to the post.
CT_CONTENT_FIELDS = [
    'Message',
    'Image Text',
    'external_link', # Note, this is an added field to the default CT data files.
    'Link Text',
]

# These are the fields to use for language prediction of CT posts
CT_LANGUAGE_FIELDS = [
    'Message',
    'Link Text',
    'Description',
]

# These are the useful fields to read in from CT data
CT_SELECT_FIELDS = [
    'Page Name',
    'User Name',
    'Facebook Id',
    'Page Admin Top Country',
    'Followers at Posting',
    'Post Created',
    'Total Interactions',
    'URL',
    'Link',
    'Message',
    'Final Link',
    'Image Text',
    'Link Text',
    'Description',
]

# Columns to output in dataframe for CT clustering
CT_OUT_COLS = [
    'cluster_id',
    'cluster_seed',
    'cluster_size',
    'cluster_score',
    'title',
    'nameid',
    'followers',
    'total_interactions',
    'num_posts',
    'coverage_within_cluster',
    'pmi_with_seed',
    'npmi_with_seed',
    'dnpmi_with_seed',
    'dnpmi_cov',
    'country',
    'name',
    'url',
]
# Columns to sort output for CT clustering
CT_SORT_COLS = [
    'cluster_score',
    'cluster_id',
    'dnpmi_with_seed',
    'coverage_within_cluster',
    'total_interactions',
    'followers',
]


# Just a function to clean up some integers in the csvs that are written as strings.
def safe_int(s):
    if isinstance(s, str):
        return int(s.replace(',', '').replace('"', ''))
    else:
        if pd.isna(s) or pd.isnull(s):
            return 0
        return int(s)


# This function is for cleaning UTM tags out of query params, which can sometimes mask otherwise identical URLs being shared
INVALID_ARG_STARTS = ['utm', 'fb']
def is_valid_query_param(s):
    for ias in INVALID_ARG_STARTS:
        if s.startswith(ias):
            return False
    return True


# This function is for cleaning UTM tags out of urls, which can sometimes mask otherwise identical URLs being shared
def normalize_url(url):
    purl = urllib.parse.urlparse(url)
    query = '&'.join([q for q in purl.query.split('&') if is_valid_query_param(q)])
    return purl.scheme.lower() + '://' + purl.netloc.lower() + purl.path + '?' + query


def ispdvalid(v):
    if not pd.isnull(v) and not pd.isna(v) and v:
        return True
    else:
        return False


def get_external_link(r):
    if ispdvalid(r['Final Link']):
        return normalize_url(r['Final Link'])
    elif ispdvalid(r['Link']) and '://www.facebook.com/' not in r['Link'].lower() and '://facebook.com/' not in r['Link'].lower():
        return normalize_url(r['Link'])
    else:
        return ''


# This function gets the relevant content out of the posts
def get_post_content(r, content_cols=CT_CONTENT_FIELDS):
    return ' '.join([r[col] for col in content_cols if ispdvalid(r[col])])


# This is just to create a clean identifier for every page. They don't all have handles, so use title of page when no handle exists.
def get_page_nameid(r):
    if not pd.isnull(r['User Name']) and not pd.isna(r['User Name']) and r['User Name']:
        return r['User Name']
    else:
        return r['Facebook Id']


# Given a list of CT data files, creates a single unified pandas dataframe
def get_ct_df(infiles, content_cols=CT_CONTENT_FIELDS):
    if isinstance(infiles, str):
        infiles = [infiles]

    df = pd.concat((pd.read_csv(f, usecols=CT_SELECT_FIELDS, dtype=str) for f in infiles))

    # Compute some additional fields to make processing easier.
    df['external_link'] = df.apply(lambda x: get_external_link(x), axis=1)
    df['total_interactions'] = df['Total Interactions'].apply(safe_int)
    df['page'] = df.apply(lambda x: get_page_nameid(x), axis=1)
    df['nameid'] = df['page']
    df['post_content'] = df.apply(lambda x: get_post_content(x, content_cols=content_cols), axis=1)
    df['post_lang_content'] = df.apply(lambda x: get_post_content(x, content_cols=CT_LANGUAGE_FIELDS), axis=1)

    return df


# Takes a dataframe from a CT file. Extracts basic page statistics into a dictionary.
# Note: dataframe needs the added 'page' field, which is a coalesce of the pages url handle and id.
def ct_extract_page_data(df):
    # Make a dict of all the page data we want. Useful later on...
    page_data = {}
    for i, r in df.iterrows():
        p = r['nameid']
        if pd.isna(p) or pd.isnull(p) or not p:
            continue
        if p not in page_data:
            page_data[p] = {
                'title': r['Page Name'],
                'name': r['User Name'],
                'followers': safe_int(r['Followers at Posting']),
                'country': r['Page Admin Top Country'],
                'id': str(r['Facebook Id']),
                'handle': r['page'],
                'nameid': p,
                'url': 'https://facebook.com/' + p,
            }
    for i, r in df.groupby('nameid')[['total_interactions']].sum().iterrows():
        page_data[i]['total_interactions'] = r['total_interactions']
    return page_data


def get_df_group_field(df, group, field):
    group_vals = {}
    for i, g in df.groupby(group):
        f = sorted(list(Counter(g[field]).items()), key=lambda x: x[1], reverse=True)
        most_f = f[0][0]
        group_vals[i] = most_f
    return group_vals


def ct_predict_page_language(df, ft_model_path):
    page_langs = {}
    df['lang0'] = df.post_lang_content.apply(lambda x: language_prediction.get_lang(x, ft_model_path))
    return get_df_group_field(df, 'nameid', 'lang0')


def ct_extract_content_clusters(df):
    # Group the posts by the content they contain. Ignore a few empty values.
    # content_clusters will be a list of lists. So for each unique piece of content (link or link + text), it will have the list of pages that shared it.
    content_groups = df.groupby(['post_content'])
    content_page_clusters = []
    for i, g in content_groups:
        if i in ('', 'This is a re-share of a post'):
            continue
        content_page_clusters.append({
            'content': i,
            'pages_sharing_content': set([u for u in g['page'] if not pd.isna(u) and not pd.isnull(u) and u])
        })

    return [list(set(l['pages_sharing_content'])) for l in content_page_clusters]


def ct_generate_page_clusters(infiles, content_cols=CT_CONTENT_FIELDS, add_language=False, ft_model_path=None, outfile=None, **kwargs):
    print("Reading in", infiles)
    df = get_ct_df(infiles, content_cols=content_cols)

    print("Computing cooccurrences...")
    content_clusters = ct_extract_content_clusters(df)
    page_data = ct_extract_page_data(df)
    co_graph = clustering.compute_cooccurrence_graph(content_clusters)

    pages = [{'page': k, 'total_interactions': v['total_interactions']} for k, v in page_data.items()]

    out_rows = []
    for page in clustering.aglomerative_cluster(co_graph, pages, len(content_clusters), 'total_interactions', 'page', **kwargs):
        if page['cluster_size'] == 1:
            continue
        page['page'] = page['node']
        page['page_url'] = 'https://facebook.com/' + page['page']
        page['num_posts'] = page['num_occurrences']
        page.update(page_data[page['page']])
        out_rows.append(page)

    print("Total number of pages in clusters:", len(out_rows))

    if not len(out_rows):
        return None

    out_cols = list(CT_OUT_COLS)
    sort_cols = list(CT_SORT_COLS)

    df_out = pd.DataFrame(out_rows)
    df_out['cluster_score'] = df_out.apply(lambda x: x['cluster_size'] * math.log10(page_data[x['cluster_seed']]['followers'] + 1), axis=1)
    df_out.sort_values(sort_cols, ascending=False, inplace=True)

    if add_language:
        print("Predicting page languages...")
        out_cols = out_cols + ['cluster_lang', 'page_lang']
        page_languages = ct_predict_page_language(df[df.nameid.isin(df_out.nameid)], ft_model_path)
        df_out['page_lang'] = df_out.nameid.apply(lambda x: page_languages.get(x, 'unknown'))
        cluster_languages = get_df_group_field(df_out, 'cluster_seed', 'page_lang')
        df_out['cluster_lang'] = df_out.cluster_seed.apply(lambda x: cluster_languages.get(x, 'unknown'))

    df_out = df_out[out_cols]

    if outfile is not None:
        df_out.to_csv(outfile, index=False)

    return df_out


if __name__ == '__main__':
    pass