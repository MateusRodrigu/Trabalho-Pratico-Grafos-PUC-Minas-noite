"""Geração de visualização HTML via PyVis.

Apesar do nome 'gephi_exporter', aqui usamos PyVis para renderizar interativamente.
"""

from pyvis.network import Network
import networkx as nx


def _edge_style(weight: int) -> tuple[str, int]:
	if weight >= 6:
		return "#00FF00", min(weight, 10)
	if weight >= 4:
		return "#FFA500", min(weight, 10)
	return "#1E90FF", max(1, min(weight, 10))


def graph_to_html(G: nx.Graph) -> str:
	net = Network(height="650px", width="100%", bgcolor="#111", font_color="white", directed=True)

	for node in G.nodes():
		net.add_node(node, label=node)

	for u, v, data in G.edges(data=True):
		peso = int(data.get("peso", 1))
		color, width = _edge_style(peso)
		net.add_edge(
			u,
			v,
			label=f"{data.get('label', '')} ({peso})",
			value=peso,
			color=color,
			width=width,
		)

	net.set_options(
		"""
		var options = {
		  "nodes": {"shape": "dot", "size": 20},
		  "edges": {"color": {"inherit": false}, "smooth": false},
		  "physics": {
			"enabled": true,
			"barnesHut": {
			  "gravitationalConstant": -80000,
			  "centralGravity": 0.2,
			  "springLength": 250,
			  "springConstant": 0.02,
			  "avoidOverlap": 1
			},
			"minVelocity": 0.75
		  }
		}
		"""
	)
	return net.generate_html("grafo.html")

