"""Camada de acesso ao Neo4j."""

from neo4j import GraphDatabase
from typing import Any, Dict, Iterable


class Neo4jService:
	def __init__(self, uri: str, user: str, password: str) -> None:
		self._driver = GraphDatabase.driver(uri, auth=(user, password))

	def close(self) -> None:
		if self._driver:
			self._driver.close()

	def query(self, cypher: str, params: Dict[str, Any] | None = None) -> Iterable[Dict[str, Any]]:
		"""Executa uma query e retorna um iterável de registros (dict-like)."""
		with self._driver.session() as session:
			result = session.run(cypher, params or {})
			for record in result:
				yield record

