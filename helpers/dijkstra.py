import numpy as np
import heapq

def create_adjacency_matrix(reward_matrix):
    rows, cols = 7, 12
    num_nodes = rows * cols
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    def node_index(r, c):
        return r * cols + c

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    for r in range(rows):
        for c in range(cols):
            current_index = node_index(r, c)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor_index = node_index(nr, nc)
                    adjacency_matrix[current_index][neighbor_index] = abs(reward_matrix[r][c])

    return adjacency_matrix

def get_best_score(reward_matrix, pt_a, pt_b):
    rows, cols = 7, 12
    point_a = tuple(pt_a[:])
    point_b = tuple(pt_b[:])
    adjacency_matrix = create_adjacency_matrix(reward_matrix)

    def node_index(r, c):
        return r * cols + c
    
    def index_to_node(index):
        return divmod(index, cols)

    start = node_index(*point_a)
    end = node_index(*point_b)
    
    # Dijkstra's Algorithm
    max_scores = [float('inf')] * (rows * cols)
    max_scores[start] = 0
    pq = [(max_scores[start], start)]  # Max-heap (negative weights for max priority)
    visited = set()

    while pq:
        current_score, current = heapq.heappop(pq)
        #current_score = -current_score

        if current in visited:
            continue
        visited.add(current)

        for neighbor in range(rows * cols):
            if adjacency_matrix[current][neighbor] > 0 and neighbor not in visited:
                weight = adjacency_matrix[current][neighbor]
                new_score = current_score + weight
                if new_score < max_scores[neighbor]:
                    max_scores[neighbor] = new_score
                    heapq.heappush(pq, (new_score, neighbor))

    path = []
    current = end
    while current != start:
        path.append(index_to_node(current))
        min_neighbor = None
        min_weight = float('inf')
        for neighbor in range(rows * cols):
            if adjacency_matrix[neighbor][current] > 0:
                if max_scores[neighbor] + adjacency_matrix[neighbor][current] == max_scores[current]:
                    if max_scores[neighbor] < min_weight:
                        min_weight = max_scores[neighbor]
                        min_neighbor = neighbor
        current = min_neighbor
        if current is None:
            break

    
    path.reverse()
    actual_score = 0
    for p in path:
        actual_score += reward_matrix[p[0]][p[1]]

    return actual_score

