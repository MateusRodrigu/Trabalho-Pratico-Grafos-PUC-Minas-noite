from app.models.abstract_graph import AbstractGraph
import csv

class AdjacencyMatrixGraph(AbstractGraph):
    def __init__(self, num_vertices: int):
        super().__init__(num_vertices)
        self.matrix = [[0.0 for _ in range(num_vertices)] for _ in range(num_vertices)]
        self.edge_count = 0

    # === Métodos obrigatórios ===
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
        return v1 == v2 and u1 != u2

    def isConvergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        return u1 == u2 and v1 != v2

    def isIncident(self, u: int, v: int, x: int) -> bool:
        return self.matrix[u][x] != 0 or self.matrix[v][x] != 0

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
        # conectividade simples: nenhum vértice isolado
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
        with open(path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Source", "Target", "Weight"])
            for i in range(self.num_vertices):
                for j in range(self.num_vertices):
                    if self.matrix[i][j] != 0:
                        writer.writerow([self.vertex_labels[i], self.vertex_labels[j], self.matrix[i][j]])
    