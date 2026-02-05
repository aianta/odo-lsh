from flask import Flask, request, jsonify, abort
from datasketch import MinHash, MinHashLSH
import networkx as nx
from networkx.readwrite import json_graph
import uuid

app = Flask(__name__)

# Simple in-memory store for stubbed endpoints
lsh_store = {}


@app.route('/minhashLSH', methods=['POST'])
def create_lsh():
    """Create a new MinHash LSH instance (stub).
    Expect JSON config in the request body. Returns created id and placeholder data.
    """
    data = request.get_json() or {}

    new_id = str(uuid.uuid4())

    # Make the MinHashLSH with the given configuration
    lsh = MinHashLSH(threshold=data['threshold'], num_perm=data['num_perm'])

    lsh_store[new_id] = {'config': data, 'lsh': lsh }
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

    print(data)

    wl_iterations = request.args.get('wl_iterations') or 16
    wl_digest_size = request.args.get('wl_digest_size') or 16
    minhash_perm = request.args.get('minhash_perm') or 256

    print(f"Got {len(data)} graphs to insert into {lsh_id}")

    minhashes = []

    for index,graph in enumerate(data):
        graph_id = graph['id'] or 'No ID'

        print(f"Computing wl-shingles for graph {graph_id}")
        # Load node-links JSON data into graph.
        G = nx.node_link_graph(graph, edges="links", directed=True, multigraph=False)

        # Compute WL-Shingles for the graph
        wl_shingles = nx.weisfeiler_lehman_subgraph_hashes(G, iterations=wl_iterations, digest_size=wl_digest_size, node_attr="color", include_initial_labels=True)

        print(f"Concatenating hashes into shingles...")
        '''
        Compute a document representing this graph as a set of WL-shingles. A document is a set of hashes representing the graph.
        wl_shingles contains hashes for each node in G. We treat each hash like a 'character', thus to produce the hashes for the graph we concatinate the hashes for each node.
        '''
        graph_document = []
        for node_hashes in wl_shingles:
            graph_document.append("".join(wl_shingles[node_hashes])) 

        print(f"Computing minhash signature for graph {graph_id}")
        # Finally, finally compute the minhash signature for this graph
        minhash = MinHash(num_perm=minhash_perm)
        for shingle in graph_document:
            minhash.update(shingle.encode('utf-8'))

        print(f"Minhash Digest for {graph_id}:\n{minhash.digest()}")

        # Append the computed minhash to a list of minhashes which we're looking to insert into the specified minhashLSH.
        minhashes.append((graph_id, minhash))
    
    
    # Retrieve the specified LSH
    lsh = lsh_store[lsh_id]
    for entry in minhashes:
        lsh.insert(entry[0], entry[1])    


    
    return jsonify({'id': lsh_id, 'message': f"Inserted {len(minhashes)} minhashes into minhashLSH {lsh_id}"}), 200


@app.route('/minhash/<lsh_id>/query', methods=['POST'])
def query_minhash(lsh_id):
    """Run a query against a MinHash index (stub).
    Accepts JSON query payload and returns placeholder results.
    """
    if lsh_id not in lsh_store:
        abort(404, description='LSH not found')
    query = request.get_json() or {}
    # TODO: implement actual query logic
    return jsonify({'id': lsh_id, 'query': query, 'results': [], 'message': 'Query endpoint stub'}), 200


@app.route('/minhashLSH/<lsh_id>/clustering', methods=['GET'])
def clustering(lsh_id):
    """Return clustering information for an LSH instance (stub).
    """
    if lsh_id not in lsh_store:
        abort(404, description='LSH not found')
    return jsonify({'id': lsh_id, 'clusters': lsh_store[lsh_id].get('clusters', {}), 'message': 'Clustering stub'}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
