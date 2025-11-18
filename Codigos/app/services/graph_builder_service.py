"""
Serviço de construção de grafos a partir de dados do Neo4j.
Responsável por criar diferentes tipos de grafos de colaboração GitHub.
"""

from typing import List, Tuple, Dict
from ..models.adjancency_list_graph import AdjacencyListGraph
from ..repositories.neo4j_repository import Neo4jRepository


class GraphBuilderService:
    """
    Serviço especializado em construir grafos a partir de dados do Neo4j.
    Suporta diferentes tipos de interações (comentários, issues, reviews, integrado).
    """

    def __init__(self, repository: Neo4jRepository):
        """
        Inicializa o serviço de construção.
        :param repository: Repositório Neo4j
        """
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
        Constrói grafo a partir de interações.
        :param interactions: Lista de tuplas (source, target, weight)
        :return: Grafo construído
        """
        # Constrói mapeamento de usuários
        self._build_user_mapping(interactions)
        
        # Cria grafo
        graph = AdjacencyListGraph(len(self.user_to_index))
        
        # Adiciona arestas
        for source, target, weight in interactions:
            u = self.user_to_index[source]
            v = self.user_to_index[target]
            
            # Se aresta já existe, soma os pesos
            if graph.hasEdge(u, v):
                current_weight = graph.getEdgeWeight(u, v)
                graph.setEdgeWeight(u, v, current_weight + weight)
            else:
                graph.addEdge(u, v)
                graph.setEdgeWeight(u, v, weight)
        
        return graph

    def build_comments_graph(self) -> AdjacencyListGraph:
        """
        Constrói grafo baseado em comentários.
        Aresta: usuário comenta em issue/PR de outro usuário.
        :return: Grafo de comentários
        """
        interactions = self.repository.fetch_comments_interactions()
        return self.build_graph_from_interactions(interactions)

    def build_issue_closure_graph(self) -> AdjacencyListGraph:
        """
        Constrói grafo baseado em fechamento de issues.
        Aresta: usuário fecha issue criada por outro usuário.
        :return: Grafo de fechamento de issues
        """
        interactions = self.repository.fetch_issue_closure_interactions()
        return self.build_graph_from_interactions(interactions)

    def build_review_graph(self) -> AdjacencyListGraph:
        """
        Constrói grafo baseado em revisões/aprovações.
        Aresta: usuário revisa/aprova/mergeia PR de outro usuário.
        :return: Grafo de revisões
        """
        interactions = self.repository.fetch_review_interactions()
        return self.build_graph_from_interactions(interactions)

    def build_integrated_graph(self) -> AdjacencyListGraph:
        """
        Constrói grafo integrado com todas as interações.
        Combina comentários, fechamento de issues e revisões.
        :return: Grafo integrado
        """
        interactions = self.repository.fetch_integrated_interactions()
        return self.build_graph_from_interactions(interactions)

    # ========================================
    # UTILIDADES
    # ========================================

    def _build_user_mapping(self, interactions: List[Tuple[str, str, int]]):
        """
        Constrói mapeamento bidirecional entre usuários e índices.
        :param interactions: Lista de interações
        """
        # Coleta usuários únicos
        unique_users = set()
        for source, target, _ in interactions:
            unique_users.add(source)
            unique_users.add(target)
        
        # Ordena para garantir consistência
        sorted_users = sorted(unique_users)
        
        # Cria mapeamentos
        self.user_to_index = {user: idx for idx, user in enumerate(sorted_users)}
        self.index_to_user = {idx: user for user, idx in self.user_to_index.items()}

    def get_all_users(self) -> List[str]:
        """
        Retorna lista de todos os usuários.
        :return: Lista de usuários
        """
        return list(self.user_to_index.keys())

    def get_user_from_index(self, index: int) -> str:
        """
        Obtém usuário a partir do índice.
        :param index: Índice do vértice
        :return: Nome do usuário
        """
        return self.index_to_user.get(index, str(index))

    def get_index_from_user(self, username: str) -> int:
        """
        Obtém índice a partir do nome do usuário.
        :param username: Nome do usuário
        :return: Índice do vértice ou -1 se não encontrado
        """
        return self.user_to_index.get(username, -1)

    def get_mapping_info(self) -> Dict:
        """
        Retorna informações sobre o mapeamento.
        :return: Dicionário com estatísticas
        """
        return {
            'total_users': len(self.user_to_index),
            'user_sample': list(self.user_to_index.keys())[:10],
            'index_range': (0, len(self.user_to_index) - 1) if self.user_to_index else (0, 0)
        }
