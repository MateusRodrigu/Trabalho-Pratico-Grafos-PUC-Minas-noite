from neo4j import GraphDatabase
from ..models.adjacency_matrix_graph import AdjacencyMatrixGraph

class GraphBuilderService:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # -------------------------------
    # MÉTODOS AUXILIARES
    # -------------------------------
    def _get_users(self):
        """Retorna todos os usuários do banco com seus IDs e logins."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User)
                RETURN u.id AS id, u.login AS login
                ORDER BY u.id
            """)
            users = [{"id": record["id"], "login": record["login"]} for record in result]
        return users

    def _add_weighted_edge(self, graph, user_index, target_index, weight):
        """Adiciona aresta ponderada, somando pesos se já existir. Ignora laços."""
        if user_index == target_index:
            # Ignora laços: grafo simples
            return
        if graph.hasEdge(user_index, target_index):
            current = graph.getEdgeWeight(user_index, target_index)
            graph.setEdgeWeight(user_index, target_index, current + weight)
        else:
            graph.addEdge(user_index, target_index)
            graph.setEdgeWeight(user_index, target_index, weight)

    # -------------------------------
    # MÉTODO PRINCIPAL
    # -------------------------------
    def build_graph_from_db(self):
        """Lê dados do Neo4j e gera o grafo consolidado ponderado."""
        users = self._get_users()
        id_to_index = {u["id"]: idx for idx, u in enumerate(users)}
        graph = AdjacencyMatrixGraph(len(users))

        with self.driver.session() as session:
            # 1️⃣ Comentários em issues/pulls (peso 2)
            query_comments = """
            MATCH (a:User)-[:COMMENTED]->(:Comment)-[:ON]->(:Issue)<-[:OPENED]-(b:User)
            RETURN DISTINCT a.id AS src, b.id AS dst
            """
            for record in session.run(query_comments):
                self._add_weighted_edge(graph, id_to_index[record["src"]], id_to_index[record["dst"]], 2)

            # 2️⃣ Fechamento de issue por outro usuário (peso 3)
            query_close = """
            MATCH (a:User)-[:CLOSED]->(:Issue)<-[:OPENED]-(b:User)
            RETURN DISTINCT a.id AS src, b.id AS dst
            """
            for record in session.run(query_close):
                self._add_weighted_edge(graph, id_to_index[record["src"]], id_to_index[record["dst"]], 3)

            # 3️⃣ Revisões/aprovações (peso 4)
            query_approve = """
            MATCH (a:User)-[:APPROVED]->(:PullRequest)<-[:OPENED]-(b:User)
            RETURN DISTINCT a.id AS src, b.id AS dst
            """
            for record in session.run(query_approve):
                self._add_weighted_edge(graph, id_to_index[record["src"]], id_to_index[record["dst"]], 4)

            # 4️⃣ Merge de pull request (peso 5)
            query_merge = """
            MATCH (a:User)-[:MERGED]->(:PullRequest)<-[:OPENED]-(b:User)
            RETURN DISTINCT a.id AS src, b.id AS dst
            """
            for record in session.run(query_merge):
                self._add_weighted_edge(graph, id_to_index[record["src"]], id_to_index[record["dst"]], 5)

        print(f"Grafo consolidado criado com {graph.getVertexCount()} vértices e {graph.getEdgeCount()} arestas.")
        return graph, users
