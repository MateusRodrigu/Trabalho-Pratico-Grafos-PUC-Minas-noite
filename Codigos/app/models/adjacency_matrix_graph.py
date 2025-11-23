from .abstract_graph import AbstractGraph

class AdjacencyMatrixGraph(AbstractGraph):
    def __init__(self, num_vertices: int):
        super().__init__(num_vertices)
        self.matrix = [[0.0 for _ in range(num_vertices)] for _ in range(num_vertices)]
        self.edge_count = 0
    def getVertexCount(self) -> int:
        return self.num_vertices

    def getEdgeCount(self) -> int:
        return self.edge_count

    def hasEdge(self, u: int, v: int) -> bool:
        self._validate_vertex(u)
        self._validate_vertex(v)
        return self.matrix[u][v] != 0

    def addEdge(self, u: int, v: int):
        self._validate_vertex(u)
        self._validate_vertex(v)
        if u == v:
            raise ValueError("Grafo simples não permite laços.")
        if not self.hasEdge(u, v):
            self.matrix[u][v] = 1.0
            self.edge_count += 1

    def removeEdge(self, u: int, v: int):
        self._validate_vertex(u)
        self._validate_vertex(v)
        if self.hasEdge(u, v):
            self.matrix[u][v] = 0.0
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
        return sum(1 for i in range(self.num_vertices) if self.matrix[i][u] != 0)

    def getVertexOutDegree(self, u: int) -> int:
        self._validate_vertex(u)
        return sum(1 for i in range(self.num_vertices) if self.matrix[u][i] != 0)

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
            raise ValueError("Aresta inexistente para atribuir peso.")
        self.matrix[u][v] = w

    def getEdgeWeight(self, u: int, v: int) -> float:
        self._validate_vertex(u)
        self._validate_vertex(v)
        return self.matrix[u][v]

    def isConnected(self) -> bool:
        for i in range(self.num_vertices):
            if self.getVertexInDegree(i) + self.getVertexOutDegree(i) == 0:
                return False
        return True

    def isEmptyGraph(self) -> bool:
        return self.edge_count == 0

    def isCompleteGraph(self) -> bool:
        expected_edges = self.num_vertices * (self.num_vertices - 1)
        return self.edge_count == expected_edges

    def exportToGEPHI(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('Source,Target,Weight\n')
            for i in range(self.num_vertices):
                for j in range(self.num_vertices):
                    if self.matrix[i][j] != 0:
                        edge_weight = self.getEdgeWeight(i, j)
                        source_label = self.vertex_labels[i] if i < len(self.vertex_labels) else str(i)
                        target_label = self.vertex_labels[j] if j < len(self.vertex_labels) else str(j)
                        f.write(f'{source_label},{target_label},{edge_weight}\n')
    