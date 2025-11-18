from typing import Dict, List, Tuple, Optional
from ..repositories.neo4j_repository import Neo4jRepository
from ..services.graph_builder_service import GraphBuilderService
from ..services.graph_analysis_service import GraphAnalysisService
from ..models.adjacency_list_graph import AdjacencyListGraph


class GraphController:
    """
    Controller principal que orquestra a lógica de negócio.
    Coordena repositories, services e fornece interface unificada.
    """

    def __init__(self):
        self.repository = Neo4jRepository()
        self.builder_service = GraphBuilderService(self.repository)
        self.current_graph: Optional[AdjacencyListGraph] = None
        self.analysis_service: Optional[GraphAnalysisService] = None
        self.graph_type: Optional[str] = None

    def close(self):
        """Fecha conexões e libera recursos."""
        self.repository.close()

    # ========================================
    # CONSTRUÇÃO DE GRAFOS
    # ========================================

    def build_graph(self, graph_type: str) -> Dict:
        """
        Constrói grafo baseado no tipo especificado.
        :param graph_type: Tipo do grafo (comments, issues, reviews, integrated)
        :return: Dicionário com informações do grafo construído
        """
        self.graph_type = graph_type.lower()
        
        if self.graph_type == "comments":
            self.current_graph = self.builder_service.build_comments_graph()
        elif self.graph_type == "issues":
            self.current_graph = self.builder_service.build_issue_closure_graph()
        elif self.graph_type == "reviews":
            self.current_graph = self.builder_service.build_review_graph()
        elif self.graph_type == "integrated":
            self.current_graph = self.builder_service.build_integrated_graph()
        else:
            raise ValueError(f"Tipo de grafo inválido: {graph_type}")
        
        # Inicializa serviço de análise
        self.analysis_service = GraphAnalysisService(
            self.current_graph,
            self.builder_service.index_to_user
        )
        
        return {
            'type': self.graph_type,
            'vertices': self.current_graph.getVertexCount(),
            'edges': self.current_graph.getEdgeCount(),
            'users': self.builder_service.get_all_users()
        }

    # ========================================
    # ANÁLISES E MÉTRICAS
    # ========================================

    def calculate_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calcula todas as métricas de centralidade.
        :return: Dicionário com todas as centralidades
        """
        self._ensure_graph_loaded()
        
        return {
            'degree': self.analysis_service.degree_centrality(),
            'betweenness': self.analysis_service.betweenness_centrality(),
            'closeness': self.analysis_service.closeness_centrality(),
            'pagerank': self.analysis_service.pagerank_centrality(),
            'eigenvector': self.analysis_service.eigenvector_centrality()
        }

    def calculate_structure_metrics(self) -> Dict:
        """
        Calcula métricas de estrutura e coesão.
        :return: Dicionário com métricas estruturais
        """
        self._ensure_graph_loaded()
        
        return {
            'density': self.analysis_service.network_density(),
            'avg_clustering': self.analysis_service.average_clustering_coefficient(),
            'assortativity': self.analysis_service.assortativity_coefficient(),
            'clustering_by_node': self.analysis_service.clustering_coefficient()
        }

    def calculate_community_metrics(self) -> Dict:
        """
        Calcula métricas de comunidade.
        :return: Dicionário com métricas de comunidade
        """
        self._ensure_graph_loaded()
        
        communities = self.analysis_service.detect_communities()
        modularity = self.analysis_service.modularity(communities)
        bridges = self.analysis_service.find_bridges(top_n=10)
        
        return {
            'communities': communities,
            'modularity': modularity,
            'bridges': bridges,
            'num_communities': len(set(communities.values()))
        }

    def get_complete_analysis(self) -> Dict:
        """
        Retorna análise completa do grafo atual.
        :return: Dicionário com todas as análises
        """
        self._ensure_graph_loaded()
        
        return {
            'summary': self.analysis_service.get_summary_statistics(),
            'centrality': self.calculate_centrality_metrics(),
            'structure': self.calculate_structure_metrics(),
            'community': self.calculate_community_metrics()
        }

    def get_top_users(self, metric: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Retorna top N usuários por métrica específica.
        :param metric: Nome da métrica (degree, betweenness, etc.)
        :param top_n: Número de usuários
        :return: Lista ordenada de usuários
        """
        self._ensure_graph_loaded()
        
        centrality_metrics = self.calculate_centrality_metrics()
        
        if metric not in centrality_metrics:
            raise ValueError(f"Métrica inválida: {metric}")
        
        return self.analysis_service.get_top_users_by_metric(
            centrality_metrics[metric], 
            top_n
        )

    # ========================================
    # EXPORTAÇÃO
    # ========================================

    def export_to_gephi(self, filepath: str) -> bool:
        """
        """
        self._ensure_graph_loaded()
        
        try:
            self.current_graph.exportToGEPHI(filepath)
            return True
        except Exception as e:
            print(f"Erro ao exportar: {e}")
            return False

    def export_analysis_to_csv(self, output_dir: str) -> Dict[str, str]:
        import os
        import csv
        
        self._ensure_graph_loaded()
        os.makedirs(output_dir, exist_ok=True)
        
        files = {}
        
        # Exporta centralidades
        centralities = self.calculate_centrality_metrics()
        for metric_name, metric_data in centralities.items():
            filepath = os.path.join(output_dir, f"centrality_{metric_name}.csv")
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['User', metric_name.capitalize()])
                for user, value in sorted(metric_data.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True):
                    writer.writerow([user, value])
            files[metric_name] = filepath
        
        # Exporta comunidades
        community_data = self.calculate_community_metrics()
        comm_filepath = os.path.join(output_dir, "communities.csv")
        with open(comm_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['User', 'Community'])
            for user, comm in sorted(community_data['communities'].items()):
                writer.writerow([user, comm])
        files['communities'] = comm_filepath
        
        return files

    # ========================================
    # QUERIES ESPECÍFICAS
    # ========================================

    def get_user_interactions(self, username: str) -> Dict:
        """
        Retorna informações de interação de um usuário específico.
        :param username: Nome do usuário
        :return: Dicionário com estatísticas do usuário
        """
        self._ensure_graph_loaded()
        
        user_idx = self.builder_service.get_index_from_user(username)
        if user_idx == -1:
            raise ValueError(f"Usuário não encontrado: {username}")
        
        centralities = self.calculate_centrality_metrics()
        
        return {
            'username': username,
            'in_degree': self.current_graph.getVertexInDegree(user_idx),
            'out_degree': self.current_graph.getVertexOutDegree(user_idx),
            'degree_centrality': centralities['degree'].get(username, 0),
            'betweenness_centrality': centralities['betweenness'].get(username, 0),
            'pagerank': centralities['pagerank'].get(username, 0)
        }

    def find_shortest_path(self, source: str, target: str) -> List[str]:
        """
        Encontra caminho mais curto entre dois usuários.
        :param source: Usuário origem
        :param target: Usuário destino
        :return: Lista de usuários no caminho
        """
        self._ensure_graph_loaded()
        
        try:
            import networkx as nx
            return nx.shortest_path(
                self.analysis_service.nx_graph, 
                source=source, 
                target=target
            )
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            raise ValueError("Um dos usuários não existe no grafo")

    # ========================================
    # HELPERS
    # ========================================

    def _ensure_graph_loaded(self):
        """Valida se há grafo carregado."""
        if self.current_graph is None or self.analysis_service is None:
            raise RuntimeError(
                "Nenhum grafo carregado. Execute build_graph() primeiro."
            )

    def get_current_graph_info(self) -> Dict:
        """
        Retorna informações básicas do grafo atual.
        :return: Dicionário com informações do grafo
        """
        if self.current_graph is None:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'type': self.graph_type,
            'vertices': self.current_graph.getVertexCount(),
            'edges': self.current_graph.getEdgeCount(),
            'is_connected': self.current_graph.isConnected(),
            'density': self.analysis_service.network_density()
        }

    # ========================================
    # ANÁLISES ESPECÍFICAS DO TRABALHO
    # ========================================

    def get_most_active_collaborators(self, top_n: int = 10) -> List[Tuple[str, Dict]]:
        """
        Identifica colaboradores mais ativos considerando múltiplas métricas.
        :param top_n: Número de colaboradores
        :return: Lista com ranking e métricas combinadas
        """
        self._ensure_graph_loaded()
        
        centralities = self.calculate_centrality_metrics()
        
        # Combina métricas com pesos
        combined_scores = {}
        for user in centralities['degree'].keys():
            score = (
                centralities['degree'].get(user, 0) * 0.3 +
                centralities['betweenness'].get(user, 0) * 0.2 +
                centralities['pagerank'].get(user, 0) * 0.3 +
                centralities['closeness'].get(user, 0) * 0.2
            )
            combined_scores[user] = {
                'combined_score': score,
                'degree': centralities['degree'].get(user, 0),
                'betweenness': centralities['betweenness'].get(user, 0),
                'pagerank': centralities['pagerank'].get(user, 0),
                'closeness': centralities['closeness'].get(user, 0)
            }
        
        return sorted(combined_scores.items(), 
                     key=lambda x: x[1]['combined_score'], 
                     reverse=True)[:top_n]

    def identify_key_maintainers(self, top_n: int = 5) -> List[Tuple[str, Dict]]:
        """
        Identifica mantenedores-chave (alto PageRank + alto out-degree).
        :param top_n: Número de mantenedores
        :return: Lista de mantenedores com métricas
        """
        self._ensure_graph_loaded()
        
        pagerank = self.analysis_service.pagerank_centrality()
        
        maintainers = {}
        for user, pr_score in pagerank.items():
            user_idx = self.builder_service.get_index_from_user(user)
            out_deg = self.current_graph.getVertexOutDegree(user_idx)
            in_deg = self.current_graph.getVertexInDegree(user_idx)
            
            # Score combinado: PageRank alto + reviews/approvals (out-degree)
            maintainer_score = pr_score * (1 + out_deg * 0.1)
            
            maintainers[user] = {
                'score': maintainer_score,
                'pagerank': pr_score,
                'out_degree': out_deg,
                'in_degree': in_deg,
                'ratio': out_deg / max(in_deg, 1)
            }
        
        return sorted(maintainers.items(), 
                     key=lambda x: x[1]['score'], 
                     reverse=True)[:top_n]

    def identify_newcomers(self, min_connections: int = 1, max_connections: int = 5) -> List[Tuple[str, Dict]]:
        """
        Identifica novatos (baixa conectividade, mas presentes).
        :param min_connections: Mínimo de conexões
        :param max_connections: Máximo de conexões
        :return: Lista de novatos com métricas
        """
        self._ensure_graph_loaded()
        
        newcomers = {}
        for user, idx in self.builder_service.user_to_index.items():
            total_degree = (self.current_graph.getVertexInDegree(idx) + 
                          self.current_graph.getVertexOutDegree(idx))
            
            if min_connections <= total_degree <= max_connections:
                newcomers[user] = {
                    'total_degree': total_degree,
                    'in_degree': self.current_graph.getVertexInDegree(idx),
                    'out_degree': self.current_graph.getVertexOutDegree(idx)
                }
        
        return sorted(newcomers.items(), 
                     key=lambda x: x[1]['total_degree'])

    def analyze_collaboration_patterns(self) -> Dict:
        """
        Analisa padrões de colaboração no repositório.
        :return: Dicionário com padrões identificados
        """
        self._ensure_graph_loaded()
        
        structure = self.calculate_structure_metrics()
        communities = self.calculate_community_metrics()
        
        # Identifica tipo de rede
        density = structure['density']
        assortativity = structure['assortativity']
        
        if density > 0.5:
            network_type = "Altamente Colaborativa"
        elif density > 0.2:
            network_type = "Moderadamente Colaborativa"
        else:
            network_type = "Colaboração Esparsa"
        
        if assortativity > 0:
            collab_pattern = "Hierárquica (core-contributors interagem entre si)"
        else:
            collab_pattern = "Descentralizada (core-contributors mentoram newcomers)"
        
        return {
            'network_type': network_type,
            'collaboration_pattern': collab_pattern,
            'density': density,
            'assortativity': assortativity,
            'num_communities': communities['num_communities'],
            'modularity': communities['modularity'],
            'clustering': structure['avg_clustering']
        }

    def compare_user_influence(self, user1: str, user2: str) -> Dict:
        """
        Compara influência entre dois usuários.
        :param user1: Primeiro usuário
        :param user2: Segundo usuário
        :return: Comparação detalhada
        """
        self._ensure_graph_loaded()
        
        info1 = self.get_user_interactions(user1)
        info2 = self.get_user_interactions(user2)
        
        return {
            'user1': user1,
            'user2': user2,
            'comparison': {
                'pagerank': {
                    'user1': info1['pagerank'],
                    'user2': info2['pagerank'],
                    'winner': user1 if info1['pagerank'] > info2['pagerank'] else user2
                },
                'betweenness': {
                    'user1': info1['betweenness_centrality'],
                    'user2': info2['betweenness_centrality'],
                    'winner': user1 if info1['betweenness_centrality'] > info2['betweenness_centrality'] else user2
                },
                'total_degree': {
                    'user1': info1['in_degree'] + info1['out_degree'],
                    'user2': info2['in_degree'] + info2['out_degree'],
                    'winner': user1 if (info1['in_degree'] + info1['out_degree']) > (info2['in_degree'] + info2['out_degree']) else user2
                }
            }
        }

    def get_collaboration_strength(self, user1: str, user2: str) -> Dict:
        """
        Analisa força da colaboração entre dois usuários.
        :param user1: Primeiro usuário
        :param user2: Segundo usuário
        :return: Métricas de colaboração direta
        """
        self._ensure_graph_loaded()
        
        idx1 = self.builder_service.get_index_from_user(user1)
        idx2 = self.builder_service.get_index_from_user(user2)
        
        if idx1 == -1 or idx2 == -1:
            raise ValueError("Um ou ambos usuários não encontrados")
        
        # Colaboração direta
        edge_1_2 = self.current_graph.getEdgeWeight(idx1, idx2) if self.current_graph.hasEdge(idx1, idx2) else 0
        edge_2_1 = self.current_graph.getEdgeWeight(idx2, idx1) if self.current_graph.hasEdge(idx2, idx1) else 0
        
        # Colaboradores em comum
        neighbors1 = set(range(self.current_graph.getVertexCount())) if idx1 < self.current_graph.getVertexCount() else set()
        neighbors1 = {v for v in neighbors1 if self.current_graph.hasEdge(idx1, v)}
        
        neighbors2 = set(range(self.current_graph.getVertexCount())) if idx2 < self.current_graph.getVertexCount() else set()
        neighbors2 = {v for v in neighbors2 if self.current_graph.hasEdge(idx2, v)}
        
        common_collaborators = neighbors1.intersection(neighbors2)
        
        return {
            'user1': user1,
            'user2': user2,
            'direct_collaboration': {
                f'{user1}_to_{user2}': edge_1_2,
                f'{user2}_to_{user1}': edge_2_1,
                'total': edge_1_2 + edge_2_1,
                'bidirectional': edge_1_2 > 0 and edge_2_1 > 0
            },
            'common_collaborators': len(common_collaborators),
            'collaboration_strength': 'Strong' if (edge_1_2 + edge_2_1) > 10 else 'Moderate' if (edge_1_2 + edge_2_1) > 5 else 'Weak'
        }

    def get_graph_statistics_report(self) -> Dict:
        """
        Gera relatório estatístico completo para o trabalho.
        :return: Relatório detalhado em formato de dicionário
        """
        self._ensure_graph_loaded()
        
        return {
            'basic_info': self.get_current_graph_info(),
            'summary': self.analysis_service.get_summary_statistics(),
            'top_collaborators': self.get_most_active_collaborators(10),
            'key_maintainers': self.identify_key_maintainers(5),
            'collaboration_patterns': self.analyze_collaboration_patterns(),
            'centrality_metrics': self.calculate_centrality_metrics(),
            'structure_metrics': self.calculate_structure_metrics(),
            'community_metrics': self.calculate_community_metrics()
        }