# import matplotlib
import argparse
import os
import pickle
import random
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from numpy import random
from optimization import update_embeddings
from optimization import update_sphere
from error import measure_penalty_error
from error import measure_radial_error
from error import total_negative_radial_error
import matplotlib.pyplot as plt


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')
    
    parser.add_argument('--dataset', nargs='?', default='karate',
                        help='Input graph name for saving files')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--line-graph', nargs='?', default='graph/graph/karate_line.edgelist',
                        help='Line graph path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--l2v-iter', default=1, type=int,
                        help='Number of iterations in Line2Vec')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--alpha', type=float, default=0.1,
                        help='alpha hyperparameter. Default is 100.')

    parser.add_argument('--beta', type=float, default=0.1,
                        help='beta hyperparameter. Default is 0.1')

    parser.add_argument('--eta', type=float, default=0.01,
                        help='eta hyperparameter. Default is 0.1')

    parser.add_argument('--gamma', type=float, default=100,
                        help='gamma hyperparameter. Default is 100')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    parser.add_argument('--scratch', dest='scratch', action='store_true',
                        help='Boolean specifying if code run starts from line graph creation process')
    parser.set_defaults(scratch=False)

    return parser.parse_args()


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def read_line_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        L = nx.read_edgelist(args.line_graph, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        L = nx.read_edgelist(args.line_graph, nodetype=int, create_using=nx.DiGraph())
        for edge in L.edges():
            L[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        L = L.to_undirected()

    return L


def seeded_vector(seed_string, vector_size):
    """Create one 'random' vector (but deterministic by seed_string)"""
    # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
    once = random.RandomState(hash(seed_string) & 0xffffffff)
    return (once.rand(vector_size) - 0.5) / vector_size


def initialize_params(embeddings, nodes, edges, neighbors, edge_map, vector_size):
    node_count = len(nodes)
    centers = np.empty((node_count, vector_size), dtype=float)
    for i in range(node_count):
        n_i = nodes[i]
        neigh_i = neighbors[n_i]
        neigh_i_count = 0
        center_i = np.zeros((1, vector_size))
        for ind in range(len(neigh_i)):
            key = (n_i, neigh_i[ind])
            if key in edge_map.keys():
                edge_index = edge_map[(n_i, neigh_i[ind])]
            else:
                edge_index = edge_map[(neigh_i[ind], n_i)]
            if edge_index in edges:
                embed_index = np.where(np.array(edges) == edge_index)[0][0]
                center_i += embeddings[embed_index]
                neigh_i_count += 1
        if neigh_i_count > 0:
            center_i = center_i / neigh_i_count
            centers[i] = center_i

    # # randomize centers vector by vector, rather than materializing a huge random matrix in RAM at once
    # for i in range(node_count):
    #     # construct deterministic seed from word AND seed argument
    #     centers[i] = seeded_vector(str(nodes[i]) + str(seed_c), vector_size)

    radius = np.empty((node_count, 1), dtype=float)
    for i in range(node_count):
        neighbors_i = neighbors[nodes[i]]
        distance_list = []
        for neigh in neighbors_i:
            neigh_node_ind = np.where(nodes == neigh)[0][0]
            distance = np.linalg.norm(
                centers[i].reshape(vector_size, 1) - centers[neigh_node_ind].reshape(vector_size, 1))
            distance_list.append(distance)
        radius[i] = np.max(np.array(distance_list))

    return centers, radius


def update_optimization_params(old_embeddings, new_embeddings, centers, radii, edge_map, nodes, edges, gamma, alpha=0.1, beta=0.1, eta=0.1):
    penalty_embeddings = update_embeddings(old_embeddings, new_embeddings, centers, radii, edge_map, nodes, edges, beta=beta, eta=eta)
    centers, radii = update_sphere(penalty_embeddings, centers, radii, edge_map, nodes, edges, alpha=alpha, beta=beta, eta=eta, gamma=gamma)
    # print("Center shape :: ", centers.shape)
    return penalty_embeddings, centers, radii


def learn_embeddings(walks, edge_map, reverse_edge_map, nodes, neighbors):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''

    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)
    # print(model.index2word)
    print('Number of walks : ', len(walks))

    # List containing edge ids, embeddings of edges are stored in this order
    edges = [int(word) for word in model.index2word]
    # print('Model index2word :: ', model.index2word)

    # Initialize params after first iteration of word2vec
    cur_embeds = model.syn0
    centers, radii = initialize_params(cur_embeds, nodes, edges, neighbors, edge_map, args.dimensions)

    # List containing penalty errors over iterations
    penalty_error_list = []
    total_negative_error_list = []
    radial_error_list = []
    total_cost_list = []

    # Hyper-parameters
    alpha = args.alpha or 2.5 #0.1
    beta = args.beta or 0.1
    eta = args.eta or 0.1
    gamma_scalar = args.gamma or 100
    gamma = [gamma_scalar]*len(radii)
    print('Initial value of hyper-parameters :: alpha = %s beta = %s eta = %s gamma = %s' % (alpha, beta, eta, gamma_scalar))

    # Boolean variable to check further update of beta
    beta_update = True

    # Start updating optimization variables using penalty method and collective homophily
    for i in range(args.l2v_iter):
        print('Iteration number %s' % (i+1))
        old_centers = centers  # For rolling back in case penalty error increases
        old_radii = radii  # For rolling back in case penalty error increases
        old_embeddings = model.syn0
        model.train(walks, total_examples=model.corpus_count)
        new_embeddings = model.syn0

        penalty_embeddings, centers, radii = update_optimization_params(old_embeddings, new_embeddings, centers, radii, reverse_edge_map, nodes, edges, gamma, alpha=alpha, beta=beta, eta=eta)
        model.syn0 = penalty_embeddings
        
        #penalty_error = beta * measure_penalty_error(penalty_embeddings, centers, radii, reverse_edge_map, nodes, edges)
        penalty_error = measure_penalty_error(penalty_embeddings, centers, radii, reverse_edge_map, nodes, edges)
        
        total_negative_error = total_negative_radial_error(radii)

        if i>10 and beta_update:
            if penalty_error >= 1.2*penalty_error_list[-1]:
                beta_update = False
                model.syn0 = old_embeddings
                centers = old_centers
                radii = old_radii
                beta /= 2
                print('Penalty Error increases significantly iteration %s, So stopped increasing beta and did the roll back' % (i+1))
                continue
        penalty_error_list.append(penalty_error)
        
        for j in range(len(radii)):
            if radii[j] < 0:
                gamma[j] *= 1.2
        total_negative_error_list.append(total_negative_error)
        
        print('At iteration = %s, Hyper-parameters eta = %s and beta = %s' % (i+1, eta, beta))
        print('Penalty error after iteration %s :: %s' %(i+1, penalty_error))
        
        #radial_error = alpha * measure_radial_error(radii)
        radial_error = measure_radial_error(radii)
        radial_error_list.append(radial_error)
        
        print('Radial error after iteration %s :: %s' %(i+1, radial_error))
        print('Negative radii error after iteration %s is %s' % (i+1,total_negative_error))
        # print('Word2Vec cost after iteration %s is :: %s' %(i+1, -model.w2v_cost))
        # total_cost = beta * penalty_error + alpha * radial_error - model.w2v_cost
        # total_cost_list.append(total_cost)
        # print('Total cost after iteration %s is %s' %(i+1, total_cost))

        if beta_update: #penalty_error > 1:
            beta *= 2
        if i>4 and (i+1)% 2 == 0:
            eta /= 2

    # print('Final embeds :: ', model.syn0)
    model.save_word2vec_format(args.output)
    return penalty_error_list, total_negative_error_list, radial_error_list, total_cost_list


def modify_edge_weights(G, epsilon=0.00001):
    degree_dict = dict(G.degree())
    total_degree = np.sum(list(degree_dict.values()))
    print("Total degree of the graph : ", total_degree)
    # print(total_degree)
    edge_weight_dict = {}
    for edge in G.edges():
        sorted_edge = tuple(sorted(edge))
        start_vertex = edge[0]
        end_vertex = edge[1]
        start_vertex_degree = degree_dict[start_vertex]
        end_vertex_degree = degree_dict[end_vertex]
        edge_weight = max(np.log(float(total_degree) / (start_vertex_degree * end_vertex_degree)) + epsilon, epsilon)
        edge_weight_dict[sorted_edge] = edge_weight
    # print(edge_weight_dict)
    return edge_weight_dict


def prepare_node_weights(G, edge_weight_dict):
    node_weight_dict = {}
    for node in G.nodes():
        weight = 0
        for neighbor in G.neighbors(node):
            if (node, neighbor) in edge_weight_dict:
                weight += edge_weight_dict[(node, neighbor)]
            else:
                weight += edge_weight_dict[(neighbor, node)]
        node_weight_dict[node] = weight
    return node_weight_dict


def build_weighted_line_graph(G, L):
    degree_dict = dict(G.degree())
    edge_weight_dict = modify_edge_weights(G)
    node_weight_dict = prepare_node_weights(G, edge_weight_dict)

    line_graph_edge_weight_dict = {}
    for line_graph_edge in L.edges():
        original_graph_edge_1 = line_graph_edge[0]
        original_graph_edge_2 = line_graph_edge[1]
        common_vertex = set(original_graph_edge_1).intersection(set(original_graph_edge_2))
        start_vertex = set(original_graph_edge_1).difference(common_vertex)
        end_vertex = set(original_graph_edge_2).difference(common_vertex)
        if len(common_vertex) == 1 and len(start_vertex) != 0 and len(end_vertex) != 0:
            common_vertex = list(common_vertex)[0]
            start_vertex = list(start_vertex)[0]
            end_vertex = list(end_vertex)[0]
        else:
            # Handle the odd case of self-loops or parallel-edges
            common_vertex = original_graph_edge_1[1]
            start_vertex = original_graph_edge_1[0]
            end_vertex = original_graph_edge_2[1]

        degree_start_vertex_edge_1 = degree_dict[start_vertex]
        degree_end_vertex_edge_1 = degree_dict[common_vertex]
        if degree_start_vertex_edge_1 == 1:
            weight_contri_src_edge_1 = 1
        else:
            weight_contri_src_edge_1 = float(degree_start_vertex_edge_1) / (
                    degree_start_vertex_edge_1 + degree_end_vertex_edge_1)

        weight_dest_edge = edge_weight_dict[original_graph_edge_2]
        weight_src_edge = edge_weight_dict[original_graph_edge_1]
        weighted_degree_common_vertex = node_weight_dict[common_vertex]
        if (weighted_degree_common_vertex - weight_src_edge) == 0:
            print('In impossible case!')
            weight_contri_dest_edge_1 = 0
        else:
            weight_contri_dest_edge_1 = float(weight_dest_edge) / (weighted_degree_common_vertex - weight_src_edge)
        line_graph_edge_weight_1 = weight_contri_src_edge_1 * weight_contri_dest_edge_1
        #     line_graph_edge_weight_dict[line_graph_edge] = line_graph_edge_weight

        degree_start_vertex_edge_2 = degree_dict[end_vertex]
        degree_end_vertex_edge_2 = degree_dict[common_vertex]
        if degree_end_vertex_edge_2 == 1:
            weight_contri_src_edge_2 = 1
        else:
            weight_contri_src_edge_2 = float(degree_start_vertex_edge_2) / (
                    degree_start_vertex_edge_2 + degree_end_vertex_edge_2)

        weight_dest_edge = edge_weight_dict[original_graph_edge_1]
        weight_src_edge = edge_weight_dict[original_graph_edge_2]
        weighted_degree_common_vertex = node_weight_dict[common_vertex]
        if (weighted_degree_common_vertex - weight_src_edge) == 0:
            print('In impossible case!')
            weight_contri_dest_edge_2 = 0
        else:
            weight_contri_dest_edge_2 = float(weight_dest_edge) / (weighted_degree_common_vertex - weight_src_edge)
        line_graph_edge_weight_2 = weight_contri_src_edge_2 * weight_contri_dest_edge_2
        line_graph_edge_weight_dict[line_graph_edge] = (line_graph_edge_weight_1 + line_graph_edge_weight_2) / 2
    # print(line_graph_edge_weight_dict)
    return line_graph_edge_weight_dict


def map_edge_to_unique_index(G):
    edge_map = {}
    reverse_edge_map = {}
    index = 0
    # print(len(G.edges()))
    for edge in G.edges():
        edge_map[edge] = index
        reverse_edge_map[index] = edge
        index += 1
    # print(edge_map)
    # Save the edge map into a pickle file
    base_path = os.path.dirname(args.input)
    print(base_path)
    edge_to_node_id_dict_filename = os.path.join(base_path, 'edge_map.pkl')
    with open(edge_to_node_id_dict_filename, 'wb') as edge_to_node_id_dict_file:
        pickle.dump(edge_map, edge_to_node_id_dict_file, pickle.HIGHEST_PROTOCOL)

    reverse_edge_to_node_id_dict_filename = os.path.join(base_path, 'reverse_edge_map.pkl')
    with open(reverse_edge_to_node_id_dict_filename, 'wb') as reverse_edge_to_node_id_dict_file:
        pickle.dump(reverse_edge_map, reverse_edge_to_node_id_dict_file, pickle.HIGHEST_PROTOCOL)
    return edge_map, reverse_edge_map


def load_edge_map():
    base_path = os.path.dirname(args.input)
    edge_to_node_id_dict_filename = os.path.join(base_path, 'edge_map.pkl')
    with open(edge_to_node_id_dict_filename, 'rb') as edge_to_node_id_dict_file:
        edge_map = pickle.load(edge_to_node_id_dict_file)

    reverse_edge_to_node_id_dict_filename = os.path.join(base_path, 'reverse_edge_map.pkl')
    with open(reverse_edge_to_node_id_dict_filename, 'rb') as reverse_edge_to_node_id_dict_file:
        reverse_edge_map = pickle.load(reverse_edge_to_node_id_dict_file)
    return edge_map, reverse_edge_map


def save_line_graph(L, edge_map, line_graph_edge_weight_dict):
    # print L.edges()
    edge_count = len(L.edges())
    line_graph_edges = list(L.edges())
    L_new = nx.Graph()
    # print sorted_edges
    for i in range(edge_count):
        edge = line_graph_edges[i]
        start_vertex = edge[0]
        end_vertex = edge[1]
        start_vertex_index_line_graph_edge = edge_map[start_vertex]
        end_vertex_index_line_graph_edge = edge_map[end_vertex]
        line_graph_edge_weight = line_graph_edge_weight_dict[edge]
        L_new.add_edge(start_vertex_index_line_graph_edge, end_vertex_index_line_graph_edge,
                       weight=line_graph_edge_weight)
    # print L_new.edges(data=True)
    # print('Line graph path : ', args.line_graph)
    nx.write_edgelist(L_new, args.line_graph, data=['weight'])


def plot_error(penalty_error_list, total_negative_error_list, radial_error_list, total_cost_list):
    do_plot = True
    if do_plot:
        plt.figure()
        plt.plot(range(1, len(penalty_error_list) + 1), penalty_error_list)
        plt.ylabel('Constraint Penalty Error')
        plt.xlabel('Iterations')
        save_path = '../embed/{}/{}_PenError.png'.format(args.dataset, args.dataset)
        plt.savefig(save_path)
        # plt.show()

        plt.figure()
        plt.plot(range(1, len(total_negative_error_list) + 1), total_negative_error_list)
        plt.ylabel('Total Negative Radii Error')
        plt.xlabel('Iterations')
        save_path = '../embed/{}/{}_NegRadiiError.png'.format(args.dataset, args.dataset)
        plt.savefig(save_path)
        # plt.show()

        plt.figure()
        plt.plot(range(1, len(radial_error_list) + 1), radial_error_list)
        plt.ylabel('Radius Square Cost')
        plt.xlabel('Iterations')
        save_path = '../embed/{}/{}_RadiiSqCost.png'.format(args.dataset, args.dataset)
        plt.savefig(save_path)
        # plt.show()

    do_total_cost_plot = False
    if do_total_cost_plot:
        plt.figure()
        plt.plot(range(1, len(total_cost_list) + 1), total_cost_list)
        plt.ylabel('Total Cost')
        plt.xlabel('Iterations')
        save_path = '../embed/{}/{}_total_cost.png'.format(args.dataset, args.dataset)
        plt.savefig(save_path)


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''

    nx_G = read_graph()
    print("Number of nodes in the original graph : ", len(nx_G.nodes()))
    print("Number of edges in the original graph : ", len(nx_G.edges()))

    if args.scratch:
        nx_L = nx.line_graph(nx_G)
        print("Number of nodes in the line graph : ", len(nx_L.nodes()))
        print("Number of edges in the line graph : ", len(nx_L.edges()))
        assert len(nx_G.edges()) == len(nx_L.nodes())

        line_graph_edge_weight_dict = build_weighted_line_graph(nx_G, nx_L)
        edge_map, reverse_edge_map = map_edge_to_unique_index(nx_G)
        # print(edge_map)
        save_line_graph(nx_L, edge_map, line_graph_edge_weight_dict)

    else:
        edge_map, reverse_edge_map = load_edge_map()

    nx_L = read_line_graph()
    L = node2vec.Graph(nx_L, args.directed, args.p, args.q)
    L.preprocess_transition_probs()
    walks = L.simulate_walks(args.num_walks, args.walk_length)

    # Prepare a dictionary of nodes and their neighbours
    nodes = np.array(nx_G.nodes())
    neighbors = {}
    for node in nodes:
        neigh_n = []
        for neigh in nx_G.neighbors(node):
            neigh_n.append(neigh)
        neighbors[node] = neigh_n

    # Learn embeddings
    penalty_error_list, total_negative_error_list, radial_error_list, total_cost_list = learn_embeddings(walks, edge_map, reverse_edge_map, nodes, neighbors)
    plot_error(penalty_error_list, total_negative_error_list, radial_error_list, total_cost_list)


if __name__ == "__main__":
    args = parse_args()
    main(args)