from flask import Flask, request, jsonify, abort
from datasketch import MinHash, MinHashLSH
from datasketch import HyperLogLogPlusPlus
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import uuid
import itertools
import requests
import string
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph

app = Flask(__name__)

# Simple in-memory store for stubbed endpoints
lsh_store = {}
graph_uri_map = {}
HOST_IP = "172.29.64.1"


@app.route('/minhashLSH', methods=['POST'])
def create_lsh():
    """Create a new MinHash LSH instance (stub).
    Expect JSON config in the request body. Returns created id and placeholder data.
    """
    data = request.get_json() or {}

    new_id = str(uuid.uuid4())

    # Make the MinHashLSH with the given configuration
    lsh = MinHashLSH(threshold=data['threshold'], num_perm=data['num_perm'])

    
    lsh_store.clear() # Prevent us from running out of RAM if we're creating a lot of these.

    lsh_store[new_id] = {'config': data, 'lsh': lsh, 'minhashes': {}, 'shingles': []}
    return jsonify({'id': new_id, 'message': 'MinHash LSH created', 'config': lsh_store[new_id]['config']}), 201


@app.route('/minhashLSH/<lsh_id>', methods=['PUT'])
def update_lsh(lsh_id):
    """
    Insert one or more graphs into an existing minhashLSH. Graphs are expected to be in node-links format. 
    See:
    https://gist.github.com/mbostock/4062045
    and
    https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_graph.html

    """
    if lsh_id not in lsh_store:
        abort(404, description='LSH not found')
    data = request.get_json() or {}
    # Merge/update config (stub behavior)

    #print(data)

    wl_iterations = request.args.get('wl_iterations') or 16
    wl_digest_size = request.args.get('wl_digest_size') or 16
    minhash_perm = request.args.get('minhash_perm') or 256

    print(f"Got {len(data)} graphs to insert into {lsh_id}")

    minhashes = []

    for index,graph in enumerate(data):
        graph_id = graph['id'] or 'No ID'

        for node in graph['nodes']: # deal with any odd characters like \xa0 in the color labels.
            node['color'] = node['color'].encode('utf-8')

        # Add the graph base_uri, if it exists to our graph uri map so we can sanity check clusterings later.
        graph_uri_map[graph['id']] = graph['baseURI']

        print(f"Computing wl-shingles for graph {graph_id}")
        # Load node-links JSON data into graph.
        G = nx.node_link_graph(graph, edges="links", directed=True, multigraph=False)

        # Compute WL-Shingles for the graph
        wl_shingles = nx.weisfeiler_lehman_subgraph_hashes(G, iterations=wl_iterations, digest_size=wl_digest_size, node_attr="color", include_initial_labels=True)

        print("G nodes:")
        print(list(G.nodes(data=True)))

        shingle_store = lsh_store[lsh_id]['shingles']

        for i,node in enumerate(wl_shingles):
            minhash = MinHash(num_perm=minhash_perm)
            for shingle in wl_shingles[node]:
                minhash.update(shingle.encode('utf-8'))
            
            # If this node has an associated robustXpath, save it alongside the minihashes and shingles.
            minihash_tuple = None
            shingle_tuple = None
            g_nodes = list(G.nodes(data=True))
            if 'robustXpath' in g_nodes[i][1]:
                minihash_tuple = (graph_id + "_" + str(i), minhash, node, g_nodes[i][1]['robustXpath'])
                shingle_tuple = (graph_id + "_" + str(i), wl_shingles[node], g_nodes[i][1]['robustXpath'])
            else:
                minihash_tuple = (graph_id + "_" + str(i), minhash, node)
                shingle_tuple = (graph_id + "_" + str(i), wl_shingles[node])

            # Append the computed minhash to a list of minhashes which we're looking to insert into the specified minhashLSH.
            minhashes.append(minihash_tuple)
            shingle_store.append(shingle_tuple)

    
    # Retrieve the specified LSH
    lsh = lsh_store[lsh_id]['lsh']
    minhash_store = lsh_store[lsh_id]['minhashes']
    for entry in minhashes:
        lsh.insert(entry[0], entry[1])
        minhash_store[entry[0]]= entry[1]    


    
    return jsonify({'id': lsh_id, 'message': f"Inserted {len(minhashes)} minhashes into minhashLSH {lsh_id}"}), 200


@app.route('/minhashLSH/<lsh_id>/query', methods=['POST'])
def query_minhash(lsh_id):
    """Run a query against a MinHash index (stub).
    Accepts JSON query payload and returns placeholder results.
    """
    if lsh_id not in lsh_store:
        abort(404, description='LSH not found')
    
    wl_iterations = request.args.get('wl_iterations') or 32
    wl_digest_size = request.args.get('wl_digest_size') or 16
    minhash_perm = request.args.get('minhash_perm') or 256

    try:
        html = request.data
        headers = {"Content-Type":"text/html"}
        response = requests.post(f'http://{HOST_IP}:8055/api/nodeLinks', headers=headers, data=html)

        response = response.json()

        for node in response['nodes']: # deal with any odd characters like \xa0 
            node['color'] = node['color'].encode('utf-8')
        
        # Load node links into graph
        G = nx.node_link_graph(response, edges="links", directed=True, multigraph=False)

        # Compute WL-Shingles for the graph
        wl_shingles = nx.weisfeiler_lehman_subgraph_hashes(G, iterations=wl_iterations, digest_size=wl_digest_size, node_attr="color", include_initial_labels=True)

        graph_document = []
        for node_hashes in wl_shingles:
            graph_document.append("".join(wl_shingles[node_hashes]))

        minhash = MinHash(num_perm=minhash_perm)
        for shingle in graph_document:
            minhash.update(shingle.encode('utf-8'))
        
        lsh = lsh_store[lsh_id]['lsh']
        result = lsh.query(minhash)

        print(result)

        # TODO: implement actual query logic
        return "lol"
        #return jsonify({'id': lsh_id, 'query': query, 'results': [], 'message': 'Query endpoint stub'}), 200
    except Exception as e:
        print(e)



@app.route('/minhashLSH/<lsh_id>/clustering', methods=['GET'])
def clustering(lsh_id):
    """Return clustering information for an LSH instance (stub).
    """
    if lsh_id not in lsh_store:
        abort(404, description='LSH not found')

    lsh = lsh_store[lsh_id]['lsh']
    documents = lsh_store[lsh_id]['minhashes']

    document_similarity = []
    for document_id in documents:
        document_minhash = documents[document_id]
        result = lsh.query(document_minhash)
        #result.append(document_id)

        document_similarity.append(result)

    edges = []
    for entry in document_similarity:
        edges.extend(list(itertools.combinations(entry, 2)))

    G = nx.Graph()
    G.add_edges_from(edges)

    # nx.draw(G)  # networkx draw()
    # plt.draw()  # pyplot draw()
    # plt.show()

    clusters = list(nx.connected_components(G))
    print(clusters)
    uri_clusters = []
    for cluster in clusters:
        items = []
        for item in cluster:
            items.append(item)
        uri_clusters.append(items)

    approx_num_clusters = len(uri_clusters)

    document_list = lsh_store[lsh_id]['shingles']
    document_labels = [x[0] for x in document_list]
    
    # Compute a sparse distance matrix using the documents in this LSH model.
    # To do this we're going to compute the pairwise jaccard distance between all 
    # cluster elements detected by the LSH model. The rest of the elements in the
    # distance matrix will be set to 1. 
    distance_matrix = np.ones((len(document_list), len(document_list)), dtype=float)

    # Set all elements on the diagonal to be 0, as the distance between a document and itself will always be 0.
    np.fill_diagonal(distance_matrix, 0.)

    # Compute pairwise jaccard distance for clusters identified by LSH.
    # TODO: Consider the option of using the approximate jaccard similarity computed from minhashes to speed things up if needed.
    for cluster in clusters:
        for pair in itertools.combinations(cluster, 2):
            print(f"Computing pairwise jaccard distance between {pair[0]} and {pair[1]}")
            # resolve indicies and shingles for 1st document in pair
            # We'll need the indicies to place the computed distance correctly in the distance matrix.
            # document_list should be a list of tuples (document_id, document_shingles) where document shingles is a list of shingles.
            A = next((index,x[1]) for index,x in enumerate(document_list) if x[0] == pair[0])
            B = next((index,x[1]) for index,x in enumerate(document_list) if x[0] == pair[1])

            index_a = A[0]
            shingles_a = set(A[1])

            index_b = B[0]
            shingles_b = set(B[1])

            
            jaccard_similarity = float(len(shingles_a.intersection(shingles_b)))/float(len(shingles_a.union(shingles_b)))
            jaccard_dist = float(1 - jaccard_similarity)
            print(f"similarity({pair[0]},{pair[1]}) = {jaccard_similarity}")
            print(f"dist({pair[0]},{pair[1]}) = {jaccard_dist}")

            print(f"Updating distance matrix at [{index_a}, {index_b}] and [{index_b}, {index_a}]")
            distance_matrix[index_a, index_b] = jaccard_dist
            distance_matrix[index_b, index_a] = jaccard_dist

    print("Distance Matrix:")
    print(distance_matrix)

    

    # For troubleshooting, save the computed distance matrix
    #np.savetxt(f"{lsh_id}-distance-matrix.csv", distance_matrix, delimiter=",")

    #clusters = linkage_based_clustering(distance_matrix, document_labels, lsh_id, approx_num_clusters, document_list)
    clusters = DBSCAN_based_clustering(distance_matrix)

    # This isn't really helpful unfortunately.
    #visualize_distance_matrix(distance_matrix, lsh_id)
   

    # Build a JSON representation of the clustering
    cluster_mapping = {}
    for idx, cluster_id in enumerate(clusters.tolist()):
        if cluster_id not in cluster_mapping:
            cluster_mapping[cluster_id] = []

        document_id = document_list[idx][0]
        cluster_item = {
            'id': document_id
        }
        
        # If this document (node) has an associated robustXpath, include it here.
        if len(document_list[idx]) == 3:
            cluster_item['robustXpath'] = document_list[idx][2]

        cluster_mapping[cluster_id].append(cluster_item)


    minhash_store = lsh_store[lsh_id]['minhashes']

    return jsonify({
        'lsh': lsh_store[lsh_id]['config'],
        'num_clusters': len(np.unique(clusters)),
        'approx_num_clusters': approx_num_clusters,
        'num_nodes': len(minhash_store),
        'clusters': cluster_mapping}),200
    #return jsonify({'id': lsh_id, 'clusters': lsh_store[lsh_id].get('clusters', {}), 'message': 'Clustering stub'}), 200

def visualize_distance_matrix(distance_matrix, lsh_id):
    dt = [('len', float)]
    _distance_matrix = distance_matrix*10
    _distance_matrix = _distance_matrix.view(dt)

    G = nx.from_numpy_array(_distance_matrix)
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), string.ascii_uppercase)))
    G = to_agraph(G)

    G.node_attr.update(color='red', style='filled')
    G.edge_attr.update(color='blue', width='2.0')

    G.draw(f"{lsh_id}-distances.png", format="png", prog="neato")

def DBSCAN_based_clustering(distance_matrix):

    clustering = DBSCAN(eps=0.9, min_samples=2, metric='precomputed').fit(distance_matrix)

    return clustering.labels_


def linkage_based_clustering(distance_matrix, labels, lsh_id, approx_num_clusters, document_list):
    # Convert our distance matrix into a condensed distance matrix, see:
    # https://stackoverflow.com/questions/60817473/warning-uncondensed-distance-matrix-in-python
    condensed_distance_matrix = squareform(distance_matrix)

    # Compute the linkage for hierarchical clustering
    linkage_data = linkage(condensed_distance_matrix, method='complete', metric='jaccard')

    # Make a dendrogram of the output
    make_dendrogram(linkage_data, labels, lsh_id)

    # Now we need to compute optimal number of clusters for the linkage data.
    # TODO In the past I've use sillhouette scoring to identify this, not sure if this is the best approach, it might be worth trying to use the HyperLogLog+ method as well.
    
    # Compute k clusterings where k ranges from 2 to the base number of clusters detected by LSH. 
    print(f"Searching for k-clusters with k from 2 to {len(document_list)}")
    clusterings = [(fcluster(linkage_data, k, criterion='maxclust'), k) for k in range(2, len(document_list))]
    #clusterings = [(fcluster(linkage_data, t, criterion='inconsistent', depth=10), t) for t in np.arange(0.1, 1.0, 0.1)]
    print(f"number of clusterings: {len(clusterings)}")

    # Only keep clusterings that partition the data into at least 2 clusters
    clusterings = [cluster for cluster in clusterings if len(np.unique(cluster[0])) > 1]
    print(f"number of clusterings with at least 2 clusters: {len(clusterings)}")

    # Compute silhouette score for each clustering
    # cluster[0] is the actual clustering -> a 1-d array of length num documents where each entry is an integer corresponding with the cluster that document belongs to. 
    # cluster[1] is k
    clusterings = [(silhouette_score(distance_matrix, labels=cluster[0], metric='precomputed'), cluster[0], cluster[1]) for cluster in clusterings]
    # Sort the clusterings by silhouette score inn descending order
    clusterings.sort(key=lambda x: x[0], reverse=True)

    # Print the silhouette scores of the clusterings
    print(f"# of documents: {len(document_list)}")
    for cluster in clusterings:
        print(f"k={cluster[2]} silhouette_score={cluster[0]}")

    # TODO handle case where there is no clustering? All documents are in the same cluster?
    
    # If there are multiple k-clusterings with the maximum silhouette score, pick the one corresponding to the smallest k.
    max_silhouette_score = clusterings[0][0]

    print(f"Max silhouette score is: {max_silhouette_score}")
    clusterings = [clustering for clustering in clusterings if clustering[0] == max_silhouette_score]
    print(f"There are {len(clusterings)} clusterings with the max silhouette score...picking the smallest number of clusters with the max silhouette score")
    clusterings.sort(key=lambda x: x[2])
    
    clusters = clusterings[0][1] # Pick the best clustering
    print(f"Picked k={clusterings[0][2]}")

    print("Clusters:")
    print(clusters)

    return clusters



def make_dendrogram(linkage_data, labels, lsh_id):
    fout = f"{lsh_id}-dendrogram.png"
    
    figure = plt.figure(0, figsize=(20,8))
    fancy_dendrogram(linkage_data, labels=labels, orientation='right', max_d=0.9999)
    figure.tight_layout()
    figure.savefig(fout)
    figure.clear()

    return fout



'''
Shamelessly copied from: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/#Eye-Candy
Then modified for horizontal dendrograms
'''
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.ylabel('sample index')
        plt.xlabel('distance')
        for i, d, c in zip( ddata['dcoord'], ddata['icoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = 0.5 * sum(d[1:3])
            if x > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % x, (x,y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axvline(x=max_d, c='k')
    return ddata

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000)


