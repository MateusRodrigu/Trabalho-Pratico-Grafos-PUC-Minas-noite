"""Controller: orquestra serviço de grafo e visualização."""

import networkx as nx
from typing import Callable, Tuple
from app.services.graph_service import GraphService


class GraphController:
	def __init__(self, graph_service: GraphService, visualizer: Callable[[nx.Graph], str]):
		self._graph_service = graph_service
		self._visualizer = visualizer

	def get_graph(self, relation_type: str) -> nx.DiGraph:
		return self._graph_service.build_graph(relation_type)

	def get_graph_html(self, relation_type: str) -> Tuple[nx.DiGraph, str]:
		G = self.get_graph(relation_type)
		if len(G.nodes) == 0:
			return G, ""  # sem html se vazio
		html = self._visualizer(G)
		return G, html

