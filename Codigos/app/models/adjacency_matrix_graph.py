import csv
from app.models.abstract_graph import AbstractGraph


class AdjacencyMatrixGraph(AbstractGraph):
    """
    Implementação de grafo direcionado simples usando matriz de adjacência.
    Compatível com a API da disciplina e com a exportação para Gephi.
    """

    def __init__(self, num_vertices: int):
        """
        Construtor da classe AdjacencyMatrixGraph.
        :param num_vertices: número de vértices do grafo
        """
        super().__init__(num_vertices)
        self.matrix = [[0 for _ in range(num_vertices)] for _ in range(num_vertices)]
        self.edge_weights = [[0.0 for _ in range(num_vertices)] for _ in range(num_vertices)]
        self.vertex_weights = [0.0 for _ in range(num_vertices)]
        self.edge_count = 0

    # ------------------------------
    # MÉTODOS OBRIGATÓRIOS DA API
    # ------------------------------

    def getVertexCount(self) -> int:
        return self.num_vertices

    def getEdgeCount(self) -> int:
        return self.edge_count

    def hasEdge(self, u: int, v: int) -> bool:
        self._validate_vertex(u)
        self._validate_vertex(v)
        return self.matrix[u][v] == 1

    def addEdge(self, u: int, v: int):
        """
        Adiciona uma aresta direcionada de u para v.
        Regras:
        - Não permite laços (u == v)
        - Não permite múltiplas arestas
        """
        self._validate_vertex(u)
        self._validate_vertex(v)
        if u == v:
            raise ValueError("Não é permitido laço em grafos simples.")
        if self.matrix[u][v] == 0:
            self.matrix[u][v] = 1
            self.edge_count += 1

    def removeEdge(self, u: int, v: int):
        self._validate_vertex(u)
        self._validate_vertex(v)
        if self.matrix[u][v] == 1:
            self.matrix[u][v] = 0
            self.edge_count -= 1

    def isSucessor(self, u: int, v: int) -> bool:
        return self.hasEdge(u, v)

    def isPredessor(self, u: int, v: int) -> bool:
        return self.hasEdge(v, u)

    def isDivergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        """Dois arcos são divergentes se partem do mesmo vértice de origem e têm destinos diferentes."""
        return u1 == u2 and v1 != v2

    def isConvergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        """Dois arcos são convergentes se chegam ao mesmo vértice destino e partem de origens diferentes."""
        return v1 == v2 and u1 != u2

    def isIncident(self, u: int, v: int, x: int) -> bool:
        """Retorna True se o vértice x é incidente à aresta (u, v)."""
        return x == u or x == v

    def getVertexInDegree(self, u: int) -> int:
        """Número de arestas que chegam em u."""
        self._validate_vertex(u)
        return sum(self.matrix[i][u] for i in range(self.num_vertices))

    def getVertexOutDegree(self, u: int) -> int:
        """Número de arestas que saem de u."""
        self._validate_vertex(u)
        return sum(self.matrix[u])

    def setVertexWeight(self, v: int, w: float):
        self._validate_vertex(v)
        self.vertex_weights[v] = w

    def getVertexWeight(self, v: int) -> float:
        self._validate_vertex(v)
        return self.vertex_weights[v]

    def setEdgeWeight(self, u: int, v: int, w: float):
        self._validate_vertex(u)
        self._validate_vertex(v)
        if self.matrix[u][v] == 0:
            raise ValueError(f"Aresta ({u}, {v}) não existe para atribuir peso.")
        self.edge_weights[u][v] = w

    def getEdgeWeight(self, u: int, v: int) -> float:
        self._validate_vertex(u)
        self._validate_vertex(v)
        return self.edge_weights[u][v]

    def isConnected(self) -> bool:
        """
        Verifica se o grafo é fortemente conexo.
        Caso haja vértices inalcançáveis, retorna False.
        """
        def dfs(start, visited):
            visited[start] = True
            for i in range(self.num_vertices):
                if self.matrix[start][i] == 1 and not visited[i]:
                    dfs(i, visited)

        for start in range(self.num_vertices):
            visited = [False] * self.num_vertices
            dfs(start, visited)
            if not all(visited):
                return False
        return True

    def isEmptyGraph(self) -> bool:
        """Retorna True se não há nenhuma aresta."""
        return self.edge_count == 0

    def isCompleteGraph(self) -> bool:
        """
        Retorna True se o grafo for direcionado e completo,
        ou seja, possui todas as arestas possíveis (u,v) com u != v.
        """
        expected_edges = self.num_vertices * (self.num_vertices - 1)
        return self.edge_count == expected_edges

    def exportToGEPHI(self, path: str):
        """
        Exporta o grafo para um arquivo CSV compatível com Gephi.
        Colunas: Source, Target, Weight
        """
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Source', 'Target', 'Weight'])
            for u in range(self.num_vertices):
                for v in range(self.num_vertices):
                    if self.matrix[u][v] == 1:
                        writer.writerow([u, v, self.edge_weights[u][v]])
