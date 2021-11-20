"""
This module contains functions, helpers, and scoring for clustering.
"""

__version__ = '0.1'
__author__ = 'Jeff Allen'


import math


# This is the default minimum threshold for clustering. The clustering value is X
CLUSTERING_MIN_THRESHOLD = 0.03
# Clustering second factor controls the limit of coverage in the 2nd best cluster, when it exists. So if the second best cluster has coverage very similar to the first best, then we assume the page is an aggregator and pull it from the cluster.
CLUSTERING_SECOND_FACTOR = 2.5


# Computes the overlap of pages. x is the "parent" page. y is the "child" page. And the function returns what fraction of posts made by y are contained in x. cg containts the cooccurence counts of the graph.
def coverage_within_parent_graph(x, y, cg):
    return cg[y].get(x, 0) * 1.0 / cg[y][y]


# Computes the pointwise mutual information between two pages.
def pmi(x, y, cg, n_docs, default_val=float('-inf')):
    if cg[y].get(x, 0) == 0:
        return default_val
    else:
        return math.log2(cg[y].get(x, 0) * n_docs * 1.0 / (cg[y][y] * cg[x][x]))


# Computes the normalized pointwise mutual information between two pages
def npmi(x, y, cg, n_docs):
    if cg[y].get(x, 0) == 0:
        return -1.0
    else:
        return math.log2(cg[y][y] * cg[x][x] * 1.0 / n_docs / n_docs) / math.log2(cg[y][x] * 1.0 / n_docs) - 1.0


# Computes the "dampened" normalized pointwise mutual information between two pages. Dampened so that if there are few occurrences of one of the pages. Dampening means bringing closer to 0. Dampening is done with exponential.
def damped_npmi(x, y, cg, n_docs, dampening_factor=4.0):
    if cg[y].get(x, 0) == 0:
        return -1.0 * math.exp(-1.0 * min(cg[x][x], cg[y][y]) / dampening_factor)

    return (math.log2(cg[y][y] * cg[x][x] * 1.0 / n_docs / n_docs) / math.log2(cg[y][x] * 1.0 / n_docs) - 1.0) * (1.0 - math.exp(-1.0 * min(cg[x][x], cg[y][y]) / dampening_factor))


# Given a list of clusters, where each cluster is a list of unique node ids, compute the cooccurence counts as a dict
def compute_cooccurrence_graph(clusters):
    # Compute cooccurrence graph
    cooccurrence_graph = {}
    for c in clusters:
        for page1 in c:
            if page1 not in cooccurrence_graph:
                cooccurrence_graph[page1] = {}
            cooccurrence_graph[page1][page1] = cooccurrence_graph[page1].get(page1, 0) + 1
            for page2 in c:
                if page1 == page2:
                    continue
                cooccurrence_graph[page1][page2] = cooccurrence_graph[page1].get(page2, 0) + 1
    return cooccurrence_graph


# Runs a basic aglomerative clustering method. Each cluster has a "seed". If a given node passes the threshold, then will be added
def agglomerative_cluster(
    graph,
    nodes,
    num_content_clusters,
    node_size_key=None,
    node_id_key=None,
    update_every=1000,
    min_threshold=CLUSTERING_MIN_THRESHOLD,
    second_cluster_factor=CLUSTERING_SECOND_FACTOR,
    **kwargs,
    ):

    if node_size_key is not None:
        nodes = sorted(nodes, key=lambda x: x[node_size_key], reverse=True)

    clusters = []
    for i, node in enumerate(nodes):
        print_update = False

        node_id = node[node_id_key]

        if i % update_every == 0:
            print(
                "[", i, "/", len(nodes), "]",
                "WORKING ON", node_id,
                "There are", len(clusters), "clusters so far",
            )
            print_update = True


        if node_id not in graph:
            if print_update:
                print("\t", "Page has no content posts I think...", node_id)
            continue

        if graph[node_id].get(node_id, 0) <= 1:
            continue

        n = node_id
        nc = graph.get(n, {})
        max_dist = 0.0
        max_cluster = None
        max_cluster_i = None

        matched_clusters = []
        val_clusters = [{
            'id': i,
            'seed': cluster[0],
            'coverage': nc.get(cluster[0], 0) * 1.0 / nc[n], # See coverage_within_parent_graph
            'dnpmi': (math.log2(nc[n] * graph[cluster[0]][cluster[0]] * 1.0 / num_content_clusters / num_content_clusters) / math.log2(nc[cluster[0]] * 1.0 / num_content_clusters) - 1.0) * (1.0 - math.exp(-1.0 * min(graph[cluster[0]][cluster[0]], nc[n]) / 4.0)), # See damped_npmi. Inline for speed.
        } for i, cluster in enumerate(clusters) if cluster[0] in nc]

        matched_clusters = [c for c in val_clusters if c['coverage'] * c['dnpmi'] >= min_threshold]

        for mc in matched_clusters:
            mc['cov_dnpmi'] = mc['coverage'] * mc['dnpmi']

        matched_clusters = sorted(matched_clusters, key=lambda x: (x['cov_dnpmi']), reverse=True)
        if len(matched_clusters):
            max_cluster = matched_clusters[0]
        else:
            max_cluster = {
                'seed': None,
                'cov_dnmpi': None,
            }

        if print_update:
            print("\t", "FOR", node, "CLOSEST WAS", max_cluster)

        if len(matched_clusters) == 1:
            clusters[matched_clusters[0]['id']].append(node_id)
            if print_update:
                print("\t", "ADDING", node_id, "TO", matched_clusters[0]['seed'])

        elif len(matched_clusters) > 1 and matched_clusters[0]['cov_dnpmi'] >= min_threshold and matched_clusters[0]['cov_dnpmi'] >= matched_clusters[1]['cov_dnpmi'] * second_cluster_factor:
            clusters[matched_clusters[0]['id']].append(node_id)
            if print_update:
                print("\t", "ADDING", node_id, "TO", matched_clusters[0]['seed'])

        elif graph[node_id][node_id] > 3:
            if print_update:
                print("\t", "CREATING NEW CLUSTER FOR", node_id)
            clusters.append([node_id])

        else:
            if print_update:
                print("\t", "NO MATCHING CLUSTER AND NOT ENOUGH TO JUSTIFY SEEDING", node_id)

    for i, cluster in enumerate(clusters):
        for node in cluster:
            yield {
                'cluster_id': i,
                'cluster_seed': cluster[0],
                'cluster_size': len(cluster),
                'node': node,
                'coverage_within_cluster': coverage_within_parent_graph(cluster[0], node, graph),
                'pmi_with_seed': pmi(cluster[0], node, graph, num_content_clusters),
                'npmi_with_seed': npmi(cluster[0], node, graph, num_content_clusters),
                'dnpmi_with_seed': damped_npmi(cluster[0], node, graph, num_content_clusters),
                'dnpmi_cov': damped_npmi(cluster[0], node, graph, num_content_clusters) * coverage_within_parent_graph(cluster[0], node, graph),
                'num_occurrences': graph.get(node, {}).get(node, 0),
            }


if __name__ == '__main__':
    pass
