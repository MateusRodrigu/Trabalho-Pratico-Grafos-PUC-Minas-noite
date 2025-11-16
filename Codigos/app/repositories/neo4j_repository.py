from neo4j import GraphDatabase
from typing import List, Dict, Tuple
from app.config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class Neo4jRepository:
    """
    Repositório responsável por todas as interações com o banco Neo4j.
    Implementa o padrão Repository para separar lógica de acesso a dados.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        """Fecha a conexão com o Neo4j."""
        self.driver.close()

    def fetch_comments_interactions(self) -> List[Tuple[str, str, int]]:
        """
        Busca interações de comentários em issues/PRs.
        :return: Lista de tuplas (source_user, target_user, weight)
        """
        query = """
        MATCH (a:User)-[:COMMENTED]->(:Comment)-[:ON]->(t)<-[:OPENED]-(b:User)
        WHERE a.login <> b.login
        RETURN a.login AS source, b.login AS target, 2 AS peso
        """
        return self._execute_query(query)

    def fetch_issue_closure_interactions(self) -> List[Tuple[str, str, int]]:
        """
        Busca interações de fechamento de issues.
        :return: Lista de tuplas (source_user, target_user, weight)
        """
        query = """
        MATCH (a:User)-[:CLOSED|:CLOSED_BY]->(i:Issue)<-[:OPENED]-(b:User)
        WHERE a.login <> b.login
        RETURN a.login AS source, b.login AS target, 3 AS peso
        """
        return self._execute_query(query)

    def fetch_review_interactions(self) -> List[Tuple[str, str, int]]:
        """
        Busca interações de revisão/aprovação/merge de PRs.
        :return: Lista de tuplas (source_user, target_user, weight)
        """
        query = """
        MATCH (a:User)-[r]->(p:PullRequest)<-[:OPENED]-(b:User)
        WHERE type(r) IN ['WROTE_REVIEW','APPROVED','MERGED'] 
              AND a.login <> b.login
        RETURN a.login AS source, b.login AS target,
            CASE type(r)
                WHEN 'WROTE_REVIEW' THEN 4
                WHEN 'APPROVED' THEN 5
                WHEN 'MERGED' THEN 6
                ELSE 1
            END AS peso
        """
        return self._execute_query(query)

    def fetch_integrated_interactions(self) -> List[Tuple[str, str, int]]:
        """
        Busca todas as interações consolidadas com pesos somados.
        :return: Lista de tuplas (source_user, target_user, total_weight)
        """
        query = """
        CALL {
            // Comentários
            MATCH (a:User)-[:COMMENTED]->(:Comment)-[:ON]->(t)<-[:OPENED]-(b:User)
            WHERE a.login <> b.login
            RETURN a.login AS source, b.login AS target, 2 AS peso
            UNION ALL
            // Fechamento
            MATCH (a:User)-[:CLOSED|:CLOSED_BY]->(i:Issue)<-[:OPENED]-(b:User)
            WHERE a.login <> b.login
            RETURN a.login AS source, b.login AS target, 3 AS peso
            UNION ALL
            // Revisões/Aprovações/Merges
            MATCH (a:User)-[r]->(p:PullRequest)<-[:OPENED]-(b:User)
            WHERE type(r) IN ['WROTE_REVIEW','APPROVED','MERGED'] 
                  AND a.login <> b.login
            RETURN a.login AS source, b.login AS target,
                CASE type(r)
                    WHEN 'WROTE_REVIEW' THEN 4
                    WHEN 'APPROVED' THEN 5
                    WHEN 'MERGED' THEN 6
                    ELSE 1
                END AS peso
        }
        RETURN source, target, sum(peso) AS peso_total
        """
        with self.driver.session() as session:
            results = session.run(query)
            return [(r["source"], r["target"], r["peso_total"]) 
                    for r in results if r["source"] and r["target"]]

    def fetch_all_users(self) -> List[str]:
        """
        Busca todos os usuários únicos no repositório.
        :return: Lista de logins de usuários
        """
        query = """
        MATCH (u:User)
        RETURN DISTINCT u.login AS login
        ORDER BY login
        """
        with self.driver.session() as session:
            results = session.run(query)
            return [r["login"] for r in results if r["login"]]

    def _execute_query(self, query: str) -> List[Tuple[str, str, int]]:
        """
        Executa uma query e retorna lista de interações.
        :param query: Query Cypher a ser executada
        :return: Lista de tuplas (source, target, weight)
        """
        with self.driver.session() as session:
            results = session.run(query)
            return [(r["source"], r["target"], r["peso"]) 
                    for r in results if r["source"] and r["target"]]