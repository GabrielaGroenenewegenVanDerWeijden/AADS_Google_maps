import time
import random
import matplotlib.pyplot as plt
import networkx as nx

def generate_random_weighted_graph(n, p):
    """Generate a random connected graph with n nodes, edge probability p, and random weights."""
    while True:
        graph = nx.gnp_random_graph(n, p)
        if nx.is_connected(graph):
            for (u, v) in graph.edges():
                graph.edges[u, v]['weight'] = random.randint(1, 10)
            return graph

def exhaustive_path(graph, start, end):
    """Find the shortest path using exhaustive search (all paths)."""
    try:
        all_paths = list(nx.all_simple_paths(graph, start, end))
        shortest_path = min(all_paths, key=lambda path: sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:])))
        shortest_path_weight = sum(graph[u][v]['weight'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
        return shortest_path, shortest_path_weight
    except nx.NetworkXNoPath:
        return None, float('inf')

def greedy_path(graph, start, end):
    """Find a path using a greedy algorithm (always take the next neighbor with the lowest edge weight)."""
    current = start
    path = [current]
    visited = set()
    visited.add(current)
    total_weight = 0

    while current != end:
        neighbors = list(graph.neighbors(current))
        unvisited_neighbors = [(n, graph[current][n]['weight']) for n in neighbors if n not in visited]
        if not unvisited_neighbors:
            return None, float('inf')  # No path

        next_node, weight = min(unvisited_neighbors, key=lambda x: x[1])
        path.append(next_node)
        visited.add(next_node)
        total_weight += weight
        current = next_node

    return path, total_weight

def run_experiment():
    sizes = range(5, 16)  # Graph sizes from 5 to 20 nodes
    times_exhaustive = []
    times_greedy = []
    paths_exhaustive = []
    paths_greedy = []
    graphs = []

    for size in sizes:
        graph = generate_random_weighted_graph(size, 0.4)
        graphs.append(graph)
        start, end = 0, size - 1

        # Run exhaustive approach
        start_time = time.time()
        path, weight = exhaustive_path(graph, start, end)
        end_time = time.time()
        times_exhaustive.append(end_time - start_time)
        paths_exhaustive.append((path if path else [], weight))

        # Run greedy approach
        start_time = time.time()
        path, weight = greedy_path(graph, start, end)
        end_time = time.time()
        times_greedy.append(end_time - start_time)
        paths_greedy.append((path if path else [], weight))

    return sizes, times_exhaustive, times_greedy, paths_exhaustive, paths_greedy, graphs

def plot_graph(graph, path_exhaustive, path_greedy, size, time_exhaustive, time_greedy, weight_exhaustive, weight_greedy):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(8, 6))
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)


    if path_greedy:
        edges_in_path_greedy = [(path_greedy[i], path_greedy[i + 1]) for i in range(len(path_greedy) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=edges_in_path_greedy, edge_color='magenta', width=2, label=f'Greedy Path (Weight: {weight_greedy})')
    if path_exhaustive:
        edges_in_path_exhaustive = [(path_exhaustive[i], path_exhaustive[i + 1]) for i in range(len(path_exhaustive) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=edges_in_path_exhaustive, edge_color='green', width=2, style='dashdot', label=f'Exhaustive Path (Weight: {weight_exhaustive})')

    plt.legend()
    plt.figtext(0.1, 0.1, f"Graph Size: {size}\nExhaustive Time: {time_exhaustive:.4f} sec, Greedy Time: {time_greedy:.4f} sec")
    plt.show()

def plot_results(sizes, times_exhaustive, times_greedy, paths_exhaustive, paths_greedy, graphs):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(sizes, times_greedy, label='Greedy Approach', color='magenta')
    plt.plot(sizes, times_exhaustive, label='Exhaustive Approach', linestyle='dashdot', color='green')
    plt.xlabel('Graph Size')
    plt.ylabel('Time (seconds)')
    plt.title('Time Comparison')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sizes, [weight for _, weight in paths_greedy], label='Greedy Approach', color='magenta')
    plt.plot(sizes, [weight for _, weight in paths_exhaustive], label='Exhaustive Approach', linestyle='dashdot', color='green')
    plt.xlabel('Graph Size')
    plt.ylabel('Path Weight')
    plt.title('Path Weight Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plotting paths on graphs
    for i, size in enumerate(sizes):
        plot_graph(graphs[i], paths_exhaustive[i][0], paths_greedy[i][0], size, times_exhaustive[i], times_greedy[i], paths_exhaustive[i][1], paths_greedy[i][1])

if __name__ == "__main__":
    sizes, times_exhaustive, times_greedy, paths_exhaustive, paths_greedy, graphs = run_experiment()
    plot_results(sizes, times_exhaustive, times_greedy, paths_exhaustive, paths_greedy, graphs)

