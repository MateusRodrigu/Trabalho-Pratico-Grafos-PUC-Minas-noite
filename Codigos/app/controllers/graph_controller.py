from app.models.adjacency_matrix_graph import AdjacencyMatrixGraph

def exemplo_grafo():
    g = AdjacencyMatrixGraph(4)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.set_edge_weight(0, 1, 2.5)
    print("Arestas:", g.get_edge_count())
    print("Conectado?", g.is_connected())
    g.export_to_gephi("grafo.csv")
