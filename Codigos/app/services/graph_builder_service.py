"""
Serviço de construção de grafos a partir de dados do Neo4j.
Responsável por criar diferentes tipos de grafos de colaboração GitHub.
"""

from typing import List, Tuple, Dict
from ..models.adjancency_list_graph import AdjacencyListGraph
from ..repositories.neo4j_repository import Neo4jRepository


class GraphBuilderService:

    def __init__(self, repository: Neo4jRepository):

        self.repository = repository
        self.user_to_index: Dict[str, int] = {}
        self.index_to_user: Dict[int, str] = {}


    def build_graph_from_interactions(
        self, 
        interactions: List[Tuple[str, str, int]]
    ) -> AdjacencyListGraph:


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

        interactions = self.repository.fetch_comments_interactions()
        return self.build_graph_from_interactions(interactions)

    def build_issue_closure_graph(self) -> AdjacencyListGraph:

        interactions = self.repository.fetch_issue_closure_interactions()
        return self.build_graph_from_interactions(interactions)

    def build_review_graph(self) -> AdjacencyListGraph:

        interactions = self.repository.fetch_review_interactions()
        return self.build_graph_from_interactions(interactions)

    def build_integrated_graph(self) -> AdjacencyListGraph:
        interactions = self.repository.fetch_integrated_interactions()
        return self.build_graph_from_interactions(interactions)


    def _build_user_mapping(self, interactions: List[Tuple[str, str, int]]):

        unique_users = set()
        for source, target, _ in interactions:
            unique_users.add(source)
            unique_users.add(target)
        
        sorted_users = sorted(unique_users)
        
        self.user_to_index = {user: idx for idx, user in enumerate(sorted_users)}
        self.index_to_user = {idx: user for user, idx in self.user_to_index.items()}

    def get_all_users(self) -> List[str]:
        return list(self.user_to_index.keys())

    def get_user_from_index(self, index: int) -> str:
        return self.index_to_user.get(index, str(index))

    def get_index_from_user(self, username: str) -> int:

        return self.user_to_index.get(username, -1)

    def get_mapping_info(self) -> Dict:

        return {
            'total_users': len(self.user_to_index),
            'user_sample': list(self.user_to_index.keys())[:10],
            'index_range': (0, len(self.user_to_index) - 1) if self.user_to_index else (0, 0)
        }
