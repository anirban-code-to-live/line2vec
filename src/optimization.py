from scipy.optimize import minimize
import numpy as np


def update_embeddings(old_embeddings, new_embeddings, centers, radii, edge_map, nodes, edges, beta=0.001, eta=0.1):
    edge_count = old_embeddings.shape[0]
    embed_dim = old_embeddings.shape[1]
    # assert edge_count == len(edge_map.keys())  # False in case of isolated edges in the original graph
    assert old_embeddings.shape[0] == new_embeddings.shape[0]
    assert old_embeddings.shape[1] == new_embeddings.shape[1]
    for i in range(edge_count):
        edge = edge_map[edges[i]]
        n_u = edge[0]
        n_v = edge[1]
        X_uv = old_embeddings[i]
        n_u_ind = np.where(nodes == n_u)
        n_v_ind = np.where(nodes == n_v)
        c_u = centers[n_u_ind]
        c_v = centers[n_v_ind]
        r_u = radii[n_u_ind]
        r_v = radii[n_v_ind]
        dX_uv = np.zeros((1, embed_dim))
        if np.linalg.norm(X_uv - c_u) > r_u:
            dX_uv += 2 * beta * (X_uv - c_u)
        if np.linalg.norm(X_uv - c_v) > r_v:
            dX_uv += 2 * beta * (X_uv - c_v)
        new_embeddings[i] = new_embeddings[i] - eta * dX_uv

    return new_embeddings


def update_sphere(embeddings, centers, radii, edge_map, nodes, edges, alpha=0.1, beta=0.1, eta=0.1, gamma=None):
    # Update radius and centers using gradients
    node_count = len(nodes)
    edge_count = embeddings.shape[0]
    embed_dim = embeddings.shape[1]
    dradii = np.zeros((node_count, 1))
    dcenters = np.zeros((node_count, embed_dim))
    for i in range(edge_count):
        edge = edge_map[edges[i]]
        n_u = edge[0]
        n_v = edge[1]
        n_u_ind = np.where(nodes == n_u)
        n_v_ind = np.where(nodes == n_v)
        X_uv = embeddings[i]
        c_u = centers[n_u_ind]
        c_v = centers[n_v_ind]
        r_u = radii[n_u_ind]
        r_v = radii[n_v_ind]
        if np.linalg.norm(X_uv - c_u) > r_u:
            dradii[n_u_ind] -= 2 * beta * r_u
            dcenters[n_u_ind] += 2 * beta * (c_u - X_uv)

        if np.linalg.norm(X_uv - c_v) > r_v:
            dradii[n_v_ind] -= 2 * beta * r_v
            dcenters[n_v_ind] += 2 * beta * (c_v - X_uv)

    for i in range(node_count):
        r_u = radii[i]
        dradii[i] += 2 * alpha * r_u
        if r_u < 0:
            dradii[i] -= gamma[i]

    radii -= eta * dradii
    centers -= eta * dcenters

    return centers, radii