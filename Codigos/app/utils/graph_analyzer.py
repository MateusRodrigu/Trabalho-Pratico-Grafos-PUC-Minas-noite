"""Utilitário para extrair métricas detalhadas de um grafo NetworkX
reutilizando a API definida em `AdjacencyListGraph` sem alterar os
arquivos originais dos modelos.

Pipeline:
1. Mapear labels dos nós (strings) para índices inteiros.
2. Preencher uma instância de `AdjacencyListGraph`.
3. Usar métodos da API para coletar métricas clássicas.
4. Retornar um dicionário estruturado para fácil exibição no Streamlit.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
import networkx as nx
from app.models.adjacency_list_graph import AdjacencyListGraph


def _build_adjacency_list_graph(G: nx.DiGraph) -> Tuple[AdjacencyListGraph, Dict[str, int], Dict[int, str]]:
    """Constrói um `AdjacencyListGraph` a partir de um NetworkX DiGraph.

    Retorna:
        (grafo_api, label->idx, idx->label)
    """
    labels = list(G.nodes())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    graph_api = AdjacencyListGraph(len(labels))
    # Algumas implementações podem não ter inicializado edge_count; garante existência.
    if not hasattr(graph_api, "edge_count"):
        graph_api.edge_count = 0

    # Adiciona arestas e pesos
    for u, v, data in G.edges(data=True):
        ui = label_to_idx[u]
        vi = label_to_idx[v]
        if u == v:
            # Ignora possível laço para manter regra de grafo simples
            continue
        if not graph_api.hasEdge(ui, vi):
            # Usa API para adicionar a aresta (incrementa edge_count se existir)
            graph_api.addEdge(ui, vi)
        peso = data.get("peso", data.get("weight", 0.0))
        try:
            graph_api.setEdgeWeight(ui, vi, float(peso))
        except ValueError:
            # Aresta pode não existir se ignorada: continuar
            pass
    return graph_api, label_to_idx, idx_to_label


def summarize_graph(G: nx.DiGraph) -> Dict[str, Any]:
    """Gera um dicionário com métricas abrangentes do grafo.

    Métricas globais: número de vértices, arestas, conexo, completo, vazio,
    graus médios, distribuição de graus.
    Métricas por vértice e lista de arestas com pesos e rótulos.
    """
    if G is None or len(G.nodes) == 0:
        return {"empty": True, "vertices": [], "edges": [], "global": {}}

    api_graph, label_to_idx, idx_to_label = _build_adjacency_list_graph(G)

    vertex_count = api_graph.getVertexCount()
    edge_count = api_graph.getEdgeCount()
    is_connected = api_graph.isConnected()
    is_empty = api_graph.isEmptyGraph()
    is_complete = api_graph.isCompleteGraph()

    vertex_rows: List[Dict[str, Any]] = []
    in_degrees = []
    out_degrees = []
    for idx in range(vertex_count):
        label = idx_to_label[idx]
        in_d = api_graph.getVertexInDegree(idx)
        out_d = api_graph.getVertexOutDegree(idx)
        in_degrees.append(in_d)
        out_degrees.append(out_d)
        # Soma de pesos de entrada e saída
        incoming_weight_sum = 0.0
        outgoing_weight_sum = 0.0
        # Percorre arestas para acumular pesos
        for u, v, data in G.in_edges(label, data=True):
            incoming_weight_sum += float(data.get("peso", data.get("weight", 0.0)))
        for u, v, data in G.out_edges(label, data=True):
            outgoing_weight_sum += float(data.get("peso", data.get("weight", 0.0)))

        vertex_rows.append(
            {
                "label": label,
                "in_degree": in_d,
                "out_degree": out_d,
                "incoming_weight_sum": incoming_weight_sum,
                "outgoing_weight_sum": outgoing_weight_sum,
            }
        )

    avg_in = sum(in_degrees) / vertex_count if vertex_count else 0.0
    avg_out = sum(out_degrees) / vertex_count if vertex_count else 0.0

    # Lista de arestas
    edge_rows: List[Dict[str, Any]] = []
    for u, v, data in G.edges(data=True):
        edge_rows.append(
            {
                "source": u,
                "target": v,
                "relation": data.get("label", ""),
                "weight": float(data.get("peso", data.get("weight", 0.0))),
            }
        )

    global_metrics = {
        "vertex_count": vertex_count,
        "edge_count": edge_count,
        "is_connected": is_connected,
        "is_empty": is_empty,
        "is_complete": is_complete,
        "average_in_degree": avg_in,
        "average_out_degree": avg_out,
        "max_in_degree": max(in_degrees) if in_degrees else 0,
        "max_out_degree": max(out_degrees) if out_degrees else 0,
        "min_in_degree": min(in_degrees) if in_degrees else 0,
        "min_out_degree": min(out_degrees) if out_degrees else 0,
    }

    return {
        "empty": False,
        "global": global_metrics,
        "vertices": vertex_rows,
        "edges": edge_rows,
    }
