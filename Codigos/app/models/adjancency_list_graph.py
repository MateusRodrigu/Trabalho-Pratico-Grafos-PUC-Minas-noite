import csv
from .abstract_graph import AbstractGraph


class AdjacencyListGraph(AbstractGraph):
    """
    Implementação de grafo direcionado simples utilizando lista de adjacência.
    Atende à especificação da disciplina, com API obrigatória completa.
    """

    def __init__(self, num_vertices: int):
        """
        Construtor da classe AdjacencyListGraph.
        :param num_vertices: número de vértices do grafo
        """
        super().__init__(num_vertices)
        self.adj_list = {i: [] for i in range(num_vertices)}  # dicionário: vértice -> lista de sucessores
        self.edge_weights = {}  # chave: (u, v), valor: peso da aresta
        self.vertex_weights = [0.0 for _ in range(num_vertices)]
        self.edge_count = 0  # <-- ADICIONE ESTA LINHA
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
        return v in self.adj_list[u]

    def addEdge(self, u: int, v: int):
        """
        Adiciona uma aresta direcionada de u para v.
        Restrições:
        - Não permite laços (u == v)
        - Não permite múltiplas arestas (idempotente)
        """
        self._validate_vertex(u)
        self._validate_vertex(v)
        if u == v:
            raise ValueError("Não é permitido laço em grafos simples.")
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
            self.edge_count += 1  # mantém contagem de arestas

    def removeEdge(self, u: int, v: int):
        if v in self.adj_list[u]:
            self.adj_list[u].remove(v)
            self.edge_count -= 1

    def isSucessor(self, u: int, v: int) -> bool:
        """Retorna True se v é sucessor direto de u."""
        return self.hasEdge(u, v)

    def isPredessor(self, u: int, v: int) -> bool:
        """Retorna True se u é predecessor direto de v."""
        return self.hasEdge(v, u)

    def isDivergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        """Dois arcos são divergentes se possuem a mesma origem (u1 == u2) e destinos diferentes."""
        return u1 == u2 and v1 != v2

    def isConvergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        """Dois arcos são convergentes se possuem o mesmo destino (v1 == v2) e origens diferentes."""
        return v1 == v2 and u1 != u2

    def isIncident(self, u: int, v: int, x: int) -> bool:
        """Retorna True se o vértice x é incidente à aresta (u, v)."""
        return x == u or x == v

    def getVertexInDegree(self, u: int) -> int:
        """Número de arestas que chegam em u."""
        self._validate_vertex(u)
        return sum(1 for adj in self.adj_list.values() if u in adj)

    def getVertexOutDegree(self, u: int) -> int:
        """Número de arestas que saem de u."""
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
        """
        Retorna True se o grafo for fortemente conexo (há caminho entre todos os pares de vértices).
        Como o grafo pode ser desconexo, retorna False se houver vértices inalcançáveis.
        """
        def dfs(start, visited):
            visited[start] = True
            for neighbor in self.adj_list[start]:
                if not visited[neighbor]:
                    dfs(neighbor, visited)

        for start in range(self.num_vertices):
            visited = [False] * self.num_vertices
            dfs(start, visited)
            if not all(visited):
                return False
        return True

    def isEmptyGraph(self) -> bool:
        """Retorna True se não há nenhuma aresta."""
        return self.getEdgeCount() == 0

    def isCompleteGraph(self) -> bool:
        """
        Retorna True se o grafo for direcionado e completo,
        ou seja, há uma aresta (u,v) para todos os pares u != v.
        """
        expected_edges = self.num_vertices * (self.num_vertices - 1)
        return self.getEdgeCount() == expected_edges

    def exportToGEPHI(self, path: str):
        """
        Exporta o grafo para CSV compatível com Gephi.
        Colunas: Source, Target, Weight
        """
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Source', 'Target', 'Weight'])
            for u, vizinhos in self.adj_list.items():
                for v in vizinhos:
                    writer.writerow([u, v, self.getEdgeWeight(u, v)])
                    
