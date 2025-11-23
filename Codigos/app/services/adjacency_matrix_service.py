from typing import List, Tuple, Dict, Set, Optional, Deque
from collections import deque, defaultdict
from ..models.adjacency_matrix_graph import AdjacencyMatrixGraph
from ..repositories.neo4j_repository import Neo4jRepository
import math


class AdjacencyMatrixService:
    def __init__(self, repository: Neo4jRepository):
        self.repository = repository
        self.user_to_index: Dict[str, int] = {}
        self.index_to_user: Dict[int, str] = {}

    def build_graph_from_interactions(
        self, 
        interactions: List[Tuple[str, str, int]]
    ) -> AdjacencyMatrixGraph:
        self._build_user_mapping(interactions)
        graph = AdjacencyMatrixGraph(len(self.user_to_index))
        
        for source, target, weight in interactions:
            u = self.user_to_index[source]
            v = self.user_to_index[target]
            
            if graph.hasEdge(u, v):
                current_weight = graph.getEdgeWeight(u, v)
                graph.setEdgeWeight(u, v, current_weight + weight)
            else:
                graph.addEdge(u, v)
                graph.setEdgeWeight(u, v, weight)
        
        return graph

    def build_comments_graph(self) -> AdjacencyMatrixGraph:
        interactions = self.repository.fetch_comments_interactions()
        return self.build_graph_from_interactions(interactions)

    def build_issues_graph(self) -> AdjacencyMatrixGraph:
        interactions = self.repository.fetch_issue_closure_interactions()
        return self.build_graph_from_interactions(interactions)

    def build_reviews_graph(self) -> AdjacencyMatrixGraph:
        interactions = self.repository.fetch_review_interactions()
        return self.build_graph_from_interactions(interactions)

    def build_integrated_graph(self) -> AdjacencyMatrixGraph:
        interactions = self.repository.fetch_integrated_interactions()
        return self.build_graph_from_interactions(interactions)

    def bfs(
        self, 
        graph: AdjacencyMatrixGraph, 
        start: int
    ) -> Dict[int, int]:
        graph._validate_vertex(start)
        
        distances = {start: 0}
        queue: Deque[int] = deque([start])
        visited = [False] * graph.num_vertices
        visited[start] = True
        
        while queue:
            u = queue.popleft()
            
            # Itera por todos os vértices verificando matriz[u][v]
            for v in range(graph.num_vertices):
                if graph.matrix[u][v] != 0 and not visited[v]:
                    visited[v] = True
                    distances[v] = distances[u] + 1
                    queue.append(v)
        
        return distances

    def dfs_iterative(
        self, 
        graph: AdjacencyMatrixGraph, 
        start: int
    ) -> List[int]:
        graph._validate_vertex(start)
        
        visited = []
        stack = [start]
        seen = [False] * graph.num_vertices
        seen[start] = True
        
        while stack:
            u = stack.pop()
            visited.append(u)
            
            # Itera por todos os vértices na ordem reversa
            for v in range(graph.num_vertices - 1, -1, -1):
                if graph.matrix[u][v] != 0 and not seen[v]:
                    seen[v] = True
                    stack.append(v)
        
        return visited

    def dfs_recursive(
        self, 
        graph: AdjacencyMatrixGraph, 
        start: int
    ) -> List[int]:
        graph._validate_vertex(start)
        
        visited_set = [False] * graph.num_vertices
        visited_list = []
        
        def dfs_helper(u):
            visited_set[u] = True
            visited_list.append(u)
            for v in range(graph.num_vertices):
                if graph.matrix[u][v] != 0 and not visited_set[v]:
                    dfs_helper(v)
        
        dfs_helper(start)
        return visited_list

    def find_shortest_path(
        self, 
        graph: AdjacencyMatrixGraph,
        start: int,
        end: int
    ) -> Optional[List[int]]:
        graph._validate_vertex(start)
        graph._validate_vertex(end)
        
        if start == end:
            return [start]
        
        visited = [False] * graph.num_vertices
        visited[start] = True
        queue: Deque[Tuple[int, List[int]]] = deque([(start, [start])])
        
        while queue:
            u, path = queue.popleft()
            
            for v in range(graph.num_vertices):
                if graph.matrix[u][v] != 0:
                    if v == end:
                        return path + [v]
                    
                    if not visited[v]:
                        visited[v] = True
                        queue.append((v, path + [v]))
        
        return None

    def dijkstra(
        self, 
        graph: AdjacencyMatrixGraph,
        start: int
    ) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        graph._validate_vertex(start)
        
        import heapq
        
        distances = {i: float('inf') for i in range(graph.num_vertices)}
        distances[start] = 0
        predecessors = {i: None for i in range(graph.num_vertices)}
        
        pq = [(0, start)]
        visited = [False] * graph.num_vertices
        
        while pq:
            dist_u, u = heapq.heappop(pq)
            
            if visited[u]:
                continue
            
            visited[u] = True
            
            # Itera pela linha da matriz
            for v in range(graph.num_vertices):
                if graph.matrix[u][v] != 0:
                    weight = graph.matrix[u][v]
                    new_dist = dist_u + weight
                    
                    if new_dist < distances[v]:
                        distances[v] = new_dist
                        predecessors[v] = u
                        heapq.heappush(pq, (new_dist, v))
        
        return distances, predecessors

    def floyd_warshall(
        self, 
        graph: AdjacencyMatrixGraph
    ) -> Tuple[List[List[float]], List[List[Optional[int]]]]:
        n = graph.num_vertices
        
        dist = [[float('inf')] * n for _ in range(n)]
        next_vertex = [[None] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
            for j in range(n):
                if graph.matrix[i][j] != 0:
                    dist[i][j] = graph.matrix[i][j]
                    next_vertex[i][j] = j
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_vertex[i][j] = next_vertex[i][k]
        
        return dist, next_vertex

    def find_strongly_connected_components(
        self, 
        graph: AdjacencyMatrixGraph
    ) -> List[Set[int]]:
        n = graph.num_vertices
        
        visited = [False] * n
        stack = []
        
        def dfs1(v):
            visited[v] = True
            for u in range(n):
                if graph.matrix[v][u] != 0 and not visited[u]:
                    dfs1(u)
            stack.append(v)
        
        for i in range(n):
            if not visited[i]:
                dfs1(i)
        
        transpose = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                transpose[j][i] = graph.matrix[i][j]
        
        visited = [False] * n
        components = []
        
        def dfs2(v, component, trans):
            visited[v] = True
            component.add(v)
            for u in range(n):
                if trans[v][u] != 0 and not visited[u]:
                    dfs2(u, component, trans)
        
        while stack:
            v = stack.pop()
            if not visited[v]:
                component = set()
                dfs2(v, component, transpose)
                components.append(component)
        
        return components

    def find_weakly_connected_components(
        self, 
        graph: AdjacencyMatrixGraph
    ) -> List[Set[int]]:
        n = graph.num_vertices
        visited = [False] * n
        components = []
        
        undirected = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if graph.matrix[i][j] != 0 or graph.matrix[j][i] != 0:
                    undirected[i][j] = max(graph.matrix[i][j], graph.matrix[j][i])
                    undirected[j][i] = undirected[i][j]
        
        def dfs_component(u, component):
            visited[u] = True
            component.add(u)
            for v in range(n):
                if undirected[u][v] != 0 and not visited[v]:
                    dfs_component(v, component)
        
        for i in range(n):
            if not visited[i]:
                component = set()
                dfs_component(i, component)
                components.append(component)
        
        return components

    def has_cycle(self, graph: AdjacencyMatrixGraph) -> bool:
        n = graph.num_vertices
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n
        
        def dfs_cycle(u):
            color[u] = GRAY
            for v in range(n):
                if graph.matrix[u][v] != 0:
                    if color[v] == GRAY:
                        return True
                    if color[v] == WHITE and dfs_cycle(v):
                        return True
            color[u] = BLACK
            return False
        
        for i in range(n):
            if color[i] == WHITE:
                if dfs_cycle(i):
                    return True
        return False

    def topological_sort(
        self, 
        graph: AdjacencyMatrixGraph
    ) -> Optional[List[int]]:
        n = graph.num_vertices
        in_degree = [0] * n
        
        for i in range(n):
            for j in range(n):
                if graph.matrix[i][j] != 0:
                    in_degree[j] += 1
        
        queue = deque([i for i in range(n) if in_degree[i] == 0])
        result = []
        
        while queue:
            u = queue.popleft()
            result.append(u)
            
            for v in range(n):
                if graph.matrix[u][v] != 0:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)
        
        return result if len(result) == n else None

    def get_k_hop_neighbors(
        self, 
        graph: AdjacencyMatrixGraph,
        vertex: int,
        k: int
    ) -> Set[int]:
        graph._validate_vertex(vertex)
        
        if k == 0:
            return {vertex}
        
        current_level = {vertex}
        visited = {vertex}
        
        for _ in range(k):
            next_level = set()
            for u in current_level:
                for v in range(graph.num_vertices):
                    if graph.matrix[u][v] != 0 and v not in visited:
                        visited.add(v)
                        next_level.add(v)
            current_level = next_level
            
            if not current_level:
                break
        
        return current_level

    def export_edge_list(
        self, 
        graph: AdjacencyMatrixGraph,
        filepath: str
    ):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Source Target Weight\n")
            for i in range(graph.num_vertices):
                for j in range(graph.num_vertices):
                    if graph.matrix[i][j] != 0:
                        f.write(f"{i} {j} {graph.matrix[i][j]}\n")

    def _build_user_mapping(self, interactions: List[Tuple[str, str, int]]):
        unique_users = set()
        for source, target, _ in interactions:
            unique_users.add(source)
            unique_users.add(target)
        
        sorted_users = sorted(unique_users)
        self.user_to_index = {user: idx for idx, user in enumerate(sorted_users)}
        self.index_to_user = {idx: user for user, idx in self.user_to_index.items()}
