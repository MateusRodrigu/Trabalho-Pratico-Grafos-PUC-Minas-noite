from typing import List, Tuple, Dict, Set, Optional, Deque
from collections import deque, defaultdict
from app.models.adjancency_list_graph import AdjacencyListGraph
from app.repositories.neo4j_repository import Neo4jRepository


class AdjacencyListService:
    """
    Serviço especializado para grafos com lista de adjacência.
    Otimizado para grafos esparsos (típico de redes sociais/GitHub).
    """

    def __init__(self, repository: Neo4jRepository):
        self.repository = repository
        self.user_to_index: Dict[str, int] = {}
        self.index_to_user: Dict[int, str] = {}

    # ========================================
    # CONSTRUÇÃO DE GRAFOS
    # ========================================

    def build_graph_from_interactions(
        self, 
        interactions: List[Tuple[str, str, int]]
    ) -> AdjacencyListGraph:
        """
        Constrói grafo a partir de interações do Neo4j.
        :param interactions: Lista de tuplas (source, target, weight)
        :return: Grafo construído
        """
        self._build_user_mapping(interactions)
        graph = AdjacencyListGraph(len(self.user_to_index))
        
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

    def build_comments_graph(self) -> AdjacencyListGraph:
        """Constrói grafo de comentários."""
        interactions = self.repository.fetch_comments_interactions()
        return self.build_graph_from_interactions(interactions)

    def build_issues_graph(self) -> AdjacencyListGraph:
        """Constrói grafo de fechamento de issues."""
        interactions = self.repository.fetch_issue_closure_interactions()
        return self.build_graph_from_interactions(interactions)

    def build_reviews_graph(self) -> AdjacencyListGraph:
        """Constrói grafo de revisões/aprovações."""
        interactions = self.repository.fetch_review_interactions()
        return self.build_graph_from_interactions(interactions)

    def build_integrated_graph(self) -> AdjacencyListGraph:
        """Constrói grafo integrado com todas as interações."""
        interactions = self.repository.fetch_integrated_interactions()
        return self.build_graph_from_interactions(interactions)

    # ========================================
    # ALGORITMOS DE BUSCA
    # ========================================

    def bfs(
        self, 
        graph: AdjacencyListGraph, 
        start: int
    ) -> Dict[int, int]:
        """
        Busca em Largura (BFS).
        :param graph: Grafo
        :param start: Vértice inicial
        :return: Dicionário {vértice: distância}
        """
        graph._validate_vertex(start)
        
        distances = {start: 0}
        queue: Deque[int] = deque([start])
        visited = {start}
        
        while queue:
            u = queue.popleft()
            
            for v in graph.adj_list[u]:
                if v not in visited:
                    visited.add(v)
                    distances[v] = distances[u] + 1
                    queue.append(v)
        
        return distances

    def dfs_iterative(
        self, 
        graph: AdjacencyListGraph, 
        start: int
    ) -> List[int]:
        """
        Busca em Profundidade (DFS) iterativa.
        :param graph: Grafo
        :param start: Vértice inicial
        :return: Lista de vértices visitados
        """
        graph._validate_vertex(start)
        
        visited = []
        stack = [start]
        seen = {start}
        
        while stack:
            u = stack.pop()
            visited.append(u)
            
            for v in reversed(graph.adj_list[u]):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        
        return visited

    def dfs_recursive(
        self, 
        graph: AdjacencyListGraph, 
        start: int
    ) -> List[int]:
        """
        Busca em Profundidade (DFS) recursiva.
        :param graph: Grafo
        :param start: Vértice inicial
        :return: Lista de vértices visitados
        """
        graph._validate_vertex(start)
        
        visited_set = set()
        visited_list = []
        
        def dfs_helper(u):
            visited_set.add(u)
            visited_list.append(u)
            for v in graph.adj_list[u]:
                if v not in visited_set:
                    dfs_helper(v)
        
        dfs_helper(start)
        return visited_list

    # ========================================
    # CAMINHOS E CONECTIVIDADE
    # ========================================

    def find_shortest_path(
        self, 
        graph: AdjacencyListGraph,
        start: int,
        end: int
    ) -> Optional[List[int]]:
        """
        Encontra caminho mais curto (BFS).
        :param graph: Grafo
        :param start: Vértice inicial
        :param end: Vértice final
        :return: Lista de vértices ou None
        """
        graph._validate_vertex(start)
        graph._validate_vertex(end)
        
        if start == end:
            return [start]
        
        visited = {start}
        queue: Deque[Tuple[int, List[int]]] = deque([(start, [start])])
        
        while queue:
            u, path = queue.popleft()
            
            for v in graph.adj_list[u]:
                if v == end:
                    return path + [v]
                
                if v not in visited:
                    visited.add(v)
                    queue.append((v, path + [v]))
        
        return None

    def find_all_simple_paths(
        self, 
        graph: AdjacencyListGraph,
        start: int,
        end: int,
        max_length: Optional[int] = None
    ) -> List[List[int]]:
        """
        Encontra todos os caminhos simples entre dois vértices.
        :param graph: Grafo
        :param start: Vértice inicial
        :param end: Vértice final
        :param max_length: Comprimento máximo
        :return: Lista de caminhos
        """
        graph._validate_vertex(start)
        graph._validate_vertex(end)
        
        paths = []
        
        def dfs_paths(u, path, visited):
            if max_length and len(path) > max_length:
                return
            
            if u == end:
                paths.append(path[:])
                return
            
            for v in graph.adj_list[u]:
                if v not in visited:
                    visited.add(v)
                    path.append(v)
                    dfs_paths(v, path, visited)
                    path.pop()
                    visited.remove(v)
        
        dfs_paths(start, [start], {start})
        return paths

    def dijkstra(
        self, 
        graph: AdjacencyListGraph,
        start: int
    ) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        """
        Algoritmo de Dijkstra para caminho mais curto ponderado.
        :param graph: Grafo
        :param start: Vértice inicial
        :return: (distâncias, predecessores)
        """
        graph._validate_vertex(start)
        
        import heapq
        
        distances = {i: float('inf') for i in range(graph.num_vertices)}
        distances[start] = 0
        predecessors = {i: None for i in range(graph.num_vertices)}
        
        pq = [(0, start)]
        visited = set()
        
        while pq:
            dist_u, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            
            visited.add(u)
            
            for v in graph.adj_list[u]:
                weight = graph.getEdgeWeight(u, v)
                new_dist = dist_u + weight
                
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    predecessors[v] = u
                    heapq.heappush(pq, (new_dist, v))
        
        return distances, predecessors

    # ========================================
    # COMPONENTES E CONECTIVIDADE
    # ========================================

    def find_strongly_connected_components(
        self, 
        graph: AdjacencyListGraph
    ) -> List[Set[int]]:
        """
        Algoritmo de Kosaraju para componentes fortemente conectados.
        :param graph: Grafo
        :return: Lista de componentes (conjuntos)
        """
        n = graph.num_vertices
        
        # Primeira DFS para preencher stack
        visited = [False] * n
        stack = []
        
        def dfs1(v):
            visited[v] = True
            for u in graph.adj_list[v]:
                if not visited[u]:
                    dfs1(u)
            stack.append(v)
        
        for i in range(n):
            if not visited[i]:
                dfs1(i)
        
        # Cria grafo transposto
        transpose = {i: [] for i in range(n)}
        for u in range(n):
            for v in graph.adj_list[u]:
                transpose[v].append(u)
        
        # Segunda DFS no transposto
        visited = [False] * n
        components = []
        
        def dfs2(v, component):
            visited[v] = True
            component.add(v)
            for u in transpose[v]:
                if not visited[u]:
                    dfs2(u, component)
        
        while stack:
            v = stack.pop()
            if not visited[v]:
                component = set()
                dfs2(v, component)
                components.append(component)
        
        return components

    def find_weakly_connected_components(
        self, 
        graph: AdjacencyListGraph
    ) -> List[Set[int]]:
        """
        Componentes fracamente conectados (ignora direção).
        :param graph: Grafo
        :return: Lista de componentes
        """
        n = graph.num_vertices
        visited = [False] * n
        components = []
        
        # Cria grafo não-direcionado
        undirected = defaultdict(list)
        for u in range(n):
            for v in graph.adj_list[u]:
                undirected[u].append(v)
                undirected[v].append(u)
        
        def dfs_component(u, component):
            visited[u] = True
            component.add(u)
            for v in undirected[u]:
                if not visited[v]:
                    dfs_component(v, component)
        
        for i in range(n):
            if not visited[i]:
                component = set()
                dfs_component(i, component)
                components.append(component)
        
        return components

    # ========================================
    # CICLOS E ORDENAÇÃO TOPOLÓGICA
    # ========================================

    def has_cycle(self, graph: AdjacencyListGraph) -> bool:
        """
        Verifica se grafo tem ciclo.
        :param graph: Grafo
        :return: True se tem ciclo
        """
        n = graph.num_vertices
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n
        
        def dfs_cycle(u):
            color[u] = GRAY
            for v in graph.adj_list[u]:
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

    def find_cycle(
        self, 
        graph: AdjacencyListGraph
    ) -> Optional[List[int]]:
        """
        Encontra um ciclo no grafo.
        :param graph: Grafo
        :return: Ciclo (lista de vértices) ou None
        """
        n = graph.num_vertices
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n
        parent = [-1] * n
        
        def dfs_find_cycle(u):
            color[u] = GRAY
            
            for v in graph.adj_list[u]:
                if color[v] == GRAY:
                    # Reconstrói ciclo
                    cycle = [v]
                    current = u
                    while current != v:
                        cycle.append(current)
                        current = parent[current]
                    cycle.append(v)
                    return cycle[::-1]
                
                if color[v] == WHITE:
                    parent[v] = u
                    result = dfs_find_cycle(v)
                    if result:
                        return result
            
            color[u] = BLACK
            return None
        
        for i in range(n):
            if color[i] == WHITE:
                cycle = dfs_find_cycle(i)
                if cycle:
                    return cycle
        return None

    def topological_sort(
        self, 
        graph: AdjacencyListGraph
    ) -> Optional[List[int]]:
        """
        Ordenação topológica (Kahn's algorithm).
        :param graph: Grafo
        :return: Lista ordenada ou None se tem ciclo
        """
        n = graph.num_vertices
        in_degree = [0] * n
        
        for u in range(n):
            for v in graph.adj_list[u]:
                in_degree[v] += 1
        
        queue = deque([i for i in range(n) if in_degree[i] == 0])
        result = []
        
        while queue:
            u = queue.popleft()
            result.append(u)
            
            for v in graph.adj_list[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        return result if len(result) == n else None

    # ========================================
    # MÉTRICAS E ANÁLISES
    # ========================================

    def get_k_hop_neighbors(
        self, 
        graph: AdjacencyListGraph,
        vertex: int,
        k: int
    ) -> Set[int]:
        """
        Vizinhos a k saltos de distância.
        :param graph: Grafo
        :param vertex: Vértice central
        :param k: Número de saltos
        :return: Conjunto de vizinhos
        """
        graph._validate_vertex(vertex)
        
        if k == 0:
            return {vertex}
        
        current_level = {vertex}
        visited = {vertex}
        
        for _ in range(k):
            next_level = set()
            for u in current_level:
                for v in graph.adj_list[u]:
                    if v not in visited:
                        visited.add(v)
                        next_level.add(v)
            current_level = next_level
            
            if not current_level:
                break
        
        return current_level

    def calculate_local_clustering_coefficient(
        self, 
        graph: AdjacencyListGraph,
        vertex: int
    ) -> float:
        """
        Coeficiente de aglomeração local.
        :param graph: Grafo
        :param vertex: Vértice
        :return: Coeficiente [0, 1]
        """
        graph._validate_vertex(vertex)
        
        neighbors = set(graph.adj_list[vertex])
        k = len(neighbors)
        
        if k < 2:
            return 0.0
        
        connections = 0
        neighbors_list = list(neighbors)
        
        for i, u in enumerate(neighbors_list):
            for v in neighbors_list[i + 1:]:
                if graph.hasEdge(u, v) or graph.hasEdge(v, u):
                    connections += 1
        
        max_connections = k * (k - 1) / 2
        return connections / max_connections if max_connections > 0 else 0.0

    def calculate_average_path_length(
        self, 
        graph: AdjacencyListGraph
    ) -> float:
        """
        Comprimento médio de caminho.
        :param graph: Grafo
        :return: Comprimento médio
        """
        n = graph.num_vertices
        total_distance = 0
        count = 0
        
        for start in range(n):
            distances = self.bfs(graph, start)
            for end, dist in distances.items():
                if start != end:
                    total_distance += dist
                    count += 1
        
        return total_distance / count if count > 0 else 0.0

    def get_graph_diameter(
        self, 
        graph: AdjacencyListGraph
    ) -> int:
        """
        Diâmetro do grafo (maior distância entre pares).
        :param graph: Grafo
        :return: Diâmetro
        """
        n = graph.num_vertices
        max_distance = 0
        
        for start in range(n):
            distances = self.bfs(graph, start)
            if distances:
                max_dist = max(distances.values())
                max_distance = max(max_distance, max_dist)
        
        return max_distance

    # ========================================
    # UTILIDADES
    # ========================================

    def get_out_neighbors_with_weights(
        self, 
        graph: AdjacencyListGraph,
        vertex: int
    ) -> List[Tuple[int, float]]:
        """
        Vizinhos de saída com pesos.
        :param graph: Grafo
        :param vertex: Vértice
        :return: Lista de tuplas (vizinho, peso)
        """
        graph._validate_vertex(vertex)
        return [(v, graph.getEdgeWeight(vertex, v)) for v in graph.adj_list[vertex]]

    def get_in_neighbors(
        self, 
        graph: AdjacencyListGraph,
        vertex: int
    ) -> List[int]:
        """
        Vizinhos de entrada (predecessores).
        :param graph: Grafo
        :param vertex: Vértice
        :return: Lista de predecessores
        """
        graph._validate_vertex(vertex)
        predecessors = []
        
        for u in range(graph.num_vertices):
            if vertex in graph.adj_list[u]:
                predecessors.append(u)
        
        return predecessors

    def get_reciprocal_edges(
        self, 
        graph: AdjacencyListGraph
    ) -> List[Tuple[int, int]]:
        """
        Arestas bidirecionais (recíprocas).
        :param graph: Grafo
        :return: Lista de pares (u, v)
        """
        reciprocal = []
        visited = set()
        
        for u in range(graph.num_vertices):
            for v in graph.adj_list[u]:
                if (u, v) not in visited and (v, u) not in visited:
                    if graph.hasEdge(v, u):
                        reciprocal.append((u, v))
                        visited.add((u, v))
                        visited.add((v, u))
        
        return reciprocal

    def export_edge_list(
        self, 
        graph: AdjacencyListGraph,
        filepath: str
    ):

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Source Target Weight\n")
            for u in range(graph.num_vertices):
                for v in graph.adj_list[u]:
                    weight = graph.getEdgeWeight(u, v)
                    f.write(f"{u} {v} {weight}\n")

    def _build_user_mapping(self, interactions: List[Tuple[str, str, int]]):

        unique_users = set()
        for source, target, _ in interactions:
            unique_users.add(source)
            unique_users.add(target)
        
        sorted_users = sorted(unique_users)
        self.user_to_index = {user: idx for idx, user in enumerate(sorted_users)}
        self.index_to_user = {idx: user for user, idx in self.user_to_index.items()}