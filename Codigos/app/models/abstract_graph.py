from abc import ABC, abstractmethod

class AbstractGraph(ABC):

    def __init__(self, num_vertices: int):
        if num_vertices <= 0:
            raise ValueError("O número de vértices deve ser positivo.")
        self.num_vertices = num_vertices
        self.vertex_labels = [str(i) for i in range(num_vertices)]
        self.vertex_weights = [0.0 for _ in range(num_vertices)]


    def _validate_vertex(self, v: int):
        if v < 0 or v >= self.num_vertices:
            raise IndexError(f"Vértice {v} fora do intervalo permitido.")


    @abstractmethod
    def getVertexCount(self) -> int:
        pass

    @abstractmethod
    def getEdgeCount(self) -> int:
        pass

    @abstractmethod
    def hasEdge(self, u: int, v: int) -> bool:
        pass

    @abstractmethod
    def addEdge(self, u: int, v: int):
        pass

    @abstractmethod
    def removeEdge(self, u: int, v: int):
        pass

    @abstractmethod
    def isSucessor(self, u: int, v: int) -> bool:
        pass

    @abstractmethod
    def isPredessor(self, u: int, v: int) -> bool:
        pass

    @abstractmethod
    def isDivergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        pass

    @abstractmethod
    def isConvergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        pass

    @abstractmethod
    def isIncident(self, u: int, v: int, x: int) -> bool:
        pass

    @abstractmethod
    def getVertexInDegree(self, u: int) -> int:
        pass

    @abstractmethod
    def getVertexOutDegree(self, u: int) -> int:
        pass

    @abstractmethod
    def setVertexWeight(self, v: int, w: float):
        pass

    @abstractmethod
    def getVertexWeight(self, v: int) -> float:
        pass

    @abstractmethod
    def setEdgeWeight(self, u: int, v: int, w: float):
        pass

    @abstractmethod
    def getEdgeWeight(self, u: int, v: int) -> float:
        pass

    @abstractmethod
    def isConnected(self) -> bool:
        pass

    @abstractmethod
    def isEmptyGraph(self) -> bool:
        pass

    @abstractmethod
    def isCompleteGraph(self) -> bool:
        pass

    @abstractmethod
    def exportToGEPHI(self, path: str):
        pass
