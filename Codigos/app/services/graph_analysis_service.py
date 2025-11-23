
from typing import Dict, List, Tuple, Set, Optional
from ..models.adjancency_list_graph import AdjacencyListGraph
import networkx as nx
from collections import defaultdict
import math


class GraphAnalysisService:


    def __init__(self, graph: AdjacencyListGraph, index_to_user: Dict[int, str]):

        self.graph = graph
        self.index_to_user = index_to_user
        self.nx_graph = self._build_networkx_graph()

    def _build_networkx_graph(self) -> nx.DiGraph:

        G = nx.DiGraph()
        
        for idx, username in self.index_to_user.items():
            G.add_node(username, index=idx)
        
        for u in range(self.graph.num_vertices):
            for v in self.graph.adj_list[u]:
                weight = self.graph.getEdgeWeight(u, v)
                u_name = self.index_to_user.get(u, str(u))
                v_name = self.index_to_user.get(v, str(v))
                G.add_edge(u_name, v_name, weight=weight)
        
        return G


    def degree_centrality(self) -> Dict[str, float]:

        centrality = {}
        n = self.graph.num_vertices
        
        for idx, username in self.index_to_user.items():
            in_deg = self.graph.getVertexInDegree(idx)
            out_deg = self.graph.getVertexOutDegree(idx)
            total_deg = in_deg + out_deg
            centrality[username] = total_deg / (2 * (n - 1)) if n > 1 else 0.0
        
        return centrality

    def betweenness_centrality(self) -> Dict[str, float]:
        try:
            return nx.betweenness_centrality(self.nx_graph, normalized=True)
        except Exception as e:
            print(f"Erro ao calcular betweenness: {e}")
            return {user: 0.0 for user in self.index_to_user.values()}

    def closeness_centrality(self) -> Dict[str, float]:

        try:
            if not nx.is_strongly_connected(self.nx_graph):
                return nx.closeness_centrality(self.nx_graph)
            return nx.closeness_centrality(self.nx_graph)
        except Exception as e:
            print(f"Erro ao calcular closeness: {e}")
            return {user: 0.0 for user in self.index_to_user.values()}

    def pagerank_centrality(self, alpha: float = 0.85, max_iter: int = 100) -> Dict[str, float]:
        
        try:
            return nx.pagerank(self.nx_graph, alpha=alpha, max_iter=max_iter, weight='weight')
        except Exception as e:
            print(f"Erro ao calcular PageRank: {e}")
            return {user: 1.0 / len(self.index_to_user) for user in self.index_to_user.values()}

    def katz_centrality(self, alpha: float = 0.1, beta: float = 1.0) -> Dict[str, float]:

        try:
            return nx.katz_centrality(self.nx_graph, alpha=alpha, beta=beta, weight='weight')
        except Exception as e:
            print(f"Erro ao calcular Katz: {e}")
            return {user: 0.0 for user in self.index_to_user.values()}

    def network_density(self) -> float:
        return nx.density(self.nx_graph)

    def average_clustering_coefficient(self) -> float:
        try:
            G_undirected = self.nx_graph.to_undirected()
            return nx.average_clustering(G_undirected)
        except Exception as e:
            print(f"Erro ao calcular clustering: {e}")
            return 0.0

    def clustering_coefficient(self) -> Dict[str, float]:
        try:
            G_undirected = self.nx_graph.to_undirected()
            return nx.clustering(G_undirected)
        except Exception as e:
            print(f"Erro ao calcular clustering por nó: {e}")
            return {user: 0.0 for user in self.index_to_user.values()}

    def transitivity(self) -> float:
        try:
            G_undirected = self.nx_graph.to_undirected()
            return nx.transitivity(G_undirected)
        except Exception as e:
            print(f"Erro ao calcular transitividade: {e}")
            return 0.0

    def assortativity_coefficient(self) -> float:
        try:
            return nx.degree_assortativity_coefficient(self.nx_graph)
        except Exception as e:
            print(f"Erro ao calcular assortatividade: {e}")
            return 0.0

    def reciprocity(self) -> float:
        try:
            return nx.reciprocity(self.nx_graph)
        except Exception as e:
            print(f"Erro ao calcular reciprocidade: {e}")
            return 0.0

    def detect_communities(self, method: str = "greedy") -> Dict[str, int]:
        try:
            G_undirected = self.nx_graph.to_undirected()
            
            if method == "greedy":
                communities_gen = nx.community.greedy_modularity_communities(G_undirected, weight='weight')
            elif method == "label_propagation":
                communities_gen = nx.community.label_propagation_communities(G_undirected)
            else:
                communities_gen = nx.community.greedy_modularity_communities(G_undirected, weight='weight')
            
            community_map = {}
            for idx, community in enumerate(communities_gen):
                for node in community:
                    community_map[node] = idx
            
            return community_map
        
        except Exception as e:
            print(f"Erro ao detectar comunidades: {e}")
            return {user: idx for idx, user in enumerate(self.index_to_user.values())}

    def modularity(self, communities: Dict[str, int]) -> float:

        try:
            G_undirected = self.nx_graph.to_undirected()
            
            # Converte dicionário para lista de conjuntos
            community_sets = defaultdict(set)
            for node, comm_id in communities.items():
                community_sets[comm_id].add(node)
            
            partition = list(community_sets.values())
            return nx.community.modularity(G_undirected, partition, weight='weight')
        
        except Exception as e:
            print(f"Erro ao calcular modularidade: {e}")
            return 0.0

    def find_bridges(self, top_n: int = 10) -> List[Tuple[str, str, float]]:
        try:
            edge_betweenness = nx.edge_betweenness_centrality(self.nx_graph, normalized=True, weight='weight')
            sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return [(u, v, score) for (u, v), score in sorted_edges]
        except Exception as e:
            print(f"Erro ao calcular pontes: {e}")
            return []

    def identify_influencers(self, top_n: int = 10) -> List[Tuple[str, Dict[str, float]]]:

        pagerank = self.pagerank_centrality()
        betweenness = self.betweenness_centrality()
        degree = self.degree_centrality()
        
        combined_scores = {}
        for user in self.index_to_user.values():
            score = (
                pagerank.get(user, 0) * 0.4 +
                betweenness.get(user, 0) * 0.3 +
                degree.get(user, 0) * 0.3
            )
            combined_scores[user] = {
                'combined_score': score,
                'pagerank': pagerank.get(user, 0),
                'betweenness': betweenness.get(user, 0),
                'degree': degree.get(user, 0)
            }
        
        return sorted(combined_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)[:top_n]

    def identify_connectors(self, top_n: int = 10) -> List[Tuple[str, float]]:

        betweenness = self.betweenness_centrality()
        return sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def identify_periphery_nodes(self, threshold: float = 0.1) -> List[str]:

        degree = self.degree_centrality()
        return [user for user, cent in degree.items() if cent < threshold]

    def calculate_core_periphery(self) -> Dict[str, str]:

        degree = self.degree_centrality()
        pagerank = self.pagerank_centrality()
        
        classification = {}
        for user in self.index_to_user.values():
            deg = degree.get(user, 0)
            pr = pagerank.get(user, 0)
            
            avg_score = (deg + pr) / 2
            classification[user] = 'core' if avg_score > 0.1 else 'periphery'
        
        return classification


    def average_shortest_path_length(self) -> Optional[float]:

        try:
            if nx.is_strongly_connected(self.nx_graph):
                return nx.average_shortest_path_length(self.nx_graph, weight='weight')
            else:

                largest_scc = max(nx.strongly_connected_components(self.nx_graph), key=len)
                subgraph = self.nx_graph.subgraph(largest_scc)
                return nx.average_shortest_path_length(subgraph, weight='weight')
        except Exception as e:
            print(f"Erro ao calcular caminho médio: {e}")
            return None

    def diameter(self) -> Optional[int]:
        try:
            if nx.is_strongly_connected(self.nx_graph):
                return nx.diameter(self.nx_graph)
            else:
                # Diâmetro do maior componente
                largest_scc = max(nx.strongly_connected_components(self.nx_graph), key=len)
                subgraph = self.nx_graph.subgraph(largest_scc)
                return nx.diameter(subgraph)
        except Exception as e:
            print(f"Erro ao calcular diâmetro: {e}")
            return None


    def get_summary_statistics(self) -> Dict:
        degree_cent = self.degree_centrality()
        in_degrees = [self.graph.getVertexInDegree(i) for i in range(self.graph.num_vertices)]
        out_degrees = [self.graph.getVertexOutDegree(i) for i in range(self.graph.num_vertices)]
        
        return {
            'num_nodes': self.graph.num_vertices,
            'num_edges': self.graph.getEdgeCount(),
            'density': self.network_density(),
            'avg_in_degree': sum(in_degrees) / len(in_degrees) if in_degrees else 0,
            'avg_out_degree': sum(out_degrees) / len(out_degrees) if out_degrees else 0,
            'max_in_degree': max(in_degrees) if in_degrees else 0,
            'max_out_degree': max(out_degrees) if out_degrees else 0,
            'avg_clustering': self.average_clustering_coefficient(),
            'transitivity': self.transitivity(),
            'assortativity': self.assortativity_coefficient(),
            'reciprocity': self.reciprocity(),
            'is_connected': self.graph.isConnected(),
            'diameter': self.diameter(),
            'avg_path_length': self.average_shortest_path_length()
        }

    def get_top_users_by_metric(
        self, 
        metric_dict: Dict[str, float], 
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        return sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # ========================================
    # ANÁLISE DE COLABORAÇÃO
    # ========================================

    def analyze_collaboration_strength(self, user1: str, user2: str) -> Dict:
        if user1 not in self.nx_graph or user2 not in self.nx_graph:
            raise ValueError("Um ou ambos usuários não encontrados")
        
        direct_1_2 = self.nx_graph.get_edge_data(user1, user2)
        direct_2_1 = self.nx_graph.get_edge_data(user2, user1)
        
        weight_1_2 = direct_1_2['weight'] if direct_1_2 else 0
        weight_2_1 = direct_2_1['weight'] if direct_2_1 else 0
        
        neighbors1 = set(self.nx_graph.neighbors(user1))
        neighbors2 = set(self.nx_graph.neighbors(user2))
        common = neighbors1.intersection(neighbors2)
        
        return {
            'direct_collaboration': {
                f'{user1}_to_{user2}': weight_1_2,
                f'{user2}_to_{user1}': weight_2_1,
                'total': weight_1_2 + weight_2_1,
                'bidirectional': weight_1_2 > 0 and weight_2_1 > 0
            },
            'common_collaborators': len(common),
            'collaboration_strength': 'Strong' if (weight_1_2 + weight_2_1) > 10 else 'Moderate' if (weight_1_2 + weight_2_1) > 5 else 'Weak'
        }

    def get_collaboration_network_type(self) -> Dict[str, str]:
        density = self.network_density()
        assortativity = self.assortativity_coefficient()
        clustering = self.average_clustering_coefficient()
        
        if density > 0.5:
            network_type = "Altamente Colaborativa"
        elif density > 0.2:
            network_type = "Moderadamente Colaborativa"
        else:
            network_type = "Colaboração Esparsa"
        
        if assortativity > 0.3:
            pattern = "Hierárquica (core-contributors interagem entre si)"
        elif assortativity < -0.3:
            pattern = "Descentralizada (core-contributors mentoram newcomers)"
        else:
            pattern = "Mista (sem padrão claro)"
        
        if clustering > 0.5:
            cohesion = "Alta (grupos coesos bem definidos)"
        elif clustering > 0.3:
            cohesion = "Moderada (alguns grupos identificáveis)"
        else:
            cohesion = "Baixa (colaboração distribuída)"
        
        return {
            'network_type': network_type,
            'collaboration_pattern': pattern,
            'cohesion': cohesion,
            'density': density,
            'assortativity': assortativity,
            'clustering': clustering
        }
