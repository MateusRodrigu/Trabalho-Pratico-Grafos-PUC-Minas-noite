"""Serviço de construção de grafos a partir de consultas Cypher."""

import networkx as nx
from .neo4j_service import Neo4jService


RELATION_QUERIES = {
	"Comentários": """
		MATCH (a:User)-[:COMMENTED]->(:Comment)-[:ON]->(t)<-[:OPENED]-(b:User)
		RETURN a.login AS source, b.login AS target, 'COMENTOU' AS rel, 2 AS peso
	""",
	"Fechamento de Issue": """
		MATCH (a:User)-[:CLOSED|:CLOSED_BY]->(i:Issue)<-[:OPENED]-(b:User)
		RETURN a.login AS source, b.login AS target, 'FECHOU' AS rel, 3 AS peso
	""",
	"Revisões/Aprovações/Merges": """
		MATCH (a:User)-[r]->(p:PullRequest)<-[:OPENED]-(b:User)
		WHERE type(r) IN ['WROTE_REVIEW','APPROVED','MERGED']
		RETURN a.login AS source, b.login AS target,
			CASE type(r)
				WHEN 'WROTE_REVIEW' THEN 'REVISOU'
				WHEN 'APPROVED' THEN 'APROVOU'
				WHEN 'MERGED' THEN 'MERGEOU'
				ELSE type(r)
			END AS rel,
			CASE type(r)
				WHEN 'WROTE_REVIEW' THEN 4
				WHEN 'APPROVED' THEN 5
				WHEN 'MERGED' THEN 6
				ELSE 1
			END AS peso
	""",
	"Integrado": """
		CALL {
			MATCH (a:User)-[:COMMENTED]->(:Comment)-[:ON]->(t)<-[:OPENED]-(b:User)
			WHERE a.login <> b.login
			RETURN a.login AS source, b.login AS target, 2 AS peso
			UNION ALL
			MATCH (a:User)-[:CLOSED|:CLOSED_BY]->(i:Issue)<-[:OPENED]-(b:User)
			WHERE a.login <> b.login
			RETURN a.login AS source, b.login AS target, 3 AS peso
			UNION ALL
			MATCH (a:User)-[r]->(p:PullRequest)<-[:OPENED]-(b:User)
			WHERE type(r) IN ['WROTE_REVIEW','APPROVED','MERGED'] AND a.login <> b.login
			RETURN a.login AS source, b.login AS target,
				CASE type(r)
					WHEN 'WROTE_REVIEW' THEN 4
					WHEN 'APPROVED' THEN 5
					WHEN 'MERGED' THEN 6
					ELSE 1
				END AS peso
		}
		RETURN source, target, 'Integrado' AS rel, sum(peso) AS peso_total
	""",
}


class GraphService:
	def __init__(self, neo4j: Neo4jService) -> None:
		self._neo4j = neo4j

	def build_graph(self, relation_type: str) -> nx.DiGraph:
		cypher = RELATION_QUERIES.get(relation_type)
		if not cypher:
			return nx.DiGraph()

		G = nx.DiGraph()
		for record in self._neo4j.query(cypher):
			source = record.get("source")
			target = record.get("target")
			if not source or not target:
				continue
			# peso pode vir como 'peso' ou 'peso_total'
			peso = record.get("peso_total", record.get("peso", 1))
			rel = record.get("rel", relation_type)
			G.add_node(source)
			G.add_node(target)
			G.add_edge(source, target, label=rel, peso=peso, weight=peso)
		return G

