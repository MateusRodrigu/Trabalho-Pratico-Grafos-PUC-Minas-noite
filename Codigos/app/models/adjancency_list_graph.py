from .abstract_graph import AbstractGraph

class AdjacencyListGraph(AbstractGraph):

    def __init__(self, num_vertices: int):
        super().__init__(num_vertices)
        self.adj_list = {i: [] for i in range(num_vertices)}
        self.edge_weights = {}
        # todos os vértices começam com peso 1.0 por padrão
        self.vertex_weights = [1.0 for _ in range(num_vertices)]
        self.edge_count = 0

    def getVertexCount(self) -> int:
        return self.num_vertices

    def getEdgeCount(self) -> int:
         return self.edge_count

    def hasEdge(self, u: int, v: int) -> bool:
        self._validate_vertex(u)
        self._validate_vertex(v)
        return v in self.adj_list[u]

    def addEdge(self, u: int, v: int):
        self._validate_vertex(u)
        self._validate_vertex(v)
        if u == v:
            raise ValueError("Não é permitido laço em grafos simples.")
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
            self.edge_count += 1

    def removeEdge(self, u: int, v: int):
        if v in self.adj_list[u]:
            self.adj_list[u].remove(v)
            self.edge_count -= 1

    def isSucessor(self, u: int, v: int) -> bool:
        return self.hasEdge(u, v)

    def isPredessor(self, u: int, v: int) -> bool:
        return self.hasEdge(v, u)

    def isDivergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        return u1 == u2 and v1 != v2

    def isConvergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        return v1 == v2 and u1 != u2

    def isIncident(self, u: int, v: int, x: int) -> bool:
        self._validate_vertex(u)
        self._validate_vertex(v)
        self._validate_vertex(x)
    
        if not self.hasEdge(u, v):
            return False
    
        return x == u or x == v

    def getVertexInDegree(self, u: int) -> int:
        self._validate_vertex(u)
        return sum(1 for adj in self.adj_list.values() if u in adj)

    def getVertexOutDegree(self, u: int) -> int:
        self._validate_vertex(u)
        return len(self.adj_list[u])

    def setVertexWeight(self, v: int, w: float):
        self._validate_vertex(v)
        self.vertex_weights[v] = w

    def getVertexWeight(self, v: int) -> float:
        self._validate_vertex(v)
        return self.vertex_weights[v]

    def setEdgeWeight(self, u: int, v: int, w: float):
        self._validate_vertex(u)
        self._validate_vertex(v)
        if not self.hasEdge(u, v):
            raise ValueError(f"Aresta ({u}, {v}) não existe para atribuir peso.")
        self.edge_weights[(u, v)] = w

    def getEdgeWeight(self, u: int, v: int) -> float:
        self._validate_vertex(u)
        self._validate_vertex(v)
        return self.edge_weights.get((u, v), 0.0)

    def isConnected(self) -> bool:
        for i in range(self.num_vertices):
            if self.getVertexInDegree(i) + self.getVertexOutDegree(i) == 0:
                return False
        return True

    def isEmptyGraph(self) -> bool:
        return self.getEdgeCount() == 0

    def isCompleteGraph(self) -> bool:
        expected_edges = self.num_vertices * (self.num_vertices - 1)
        return self.getEdgeCount() == expected_edges

    def exportToGEPHI(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('Source,Target,Weight\n')
            for u, vizinhos in self.adj_list.items():
                for v in vizinhos:
                    edge_weight = self.getEdgeWeight(u, v)
                    source_label = self.vertex_labels[u] if u < len(self.vertex_labels) else str(u)
                    target_label = self.vertex_labels[v] if v < len(self.vertex_labels) else str(v)
                    f.write(f'{source_label},{target_label},{edge_weight}\n')
                    
