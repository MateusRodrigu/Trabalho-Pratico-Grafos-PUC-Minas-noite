from app.models.adjacency_matrix_graph import AdjacencyMatrixGraph

def gerar_grafo_exemplo():
    # cria um grafo de 4 vértices
    g = AdjacencyMatrixGraph(4)

    # adiciona algumas arestas
    g.addEdge(0, 1)
    g.addEdge(1, 2)
    g.addEdge(2, 3)
    g.setEdgeWeight(0, 1, 2.5)

    # mostra no terminal (opcional)
    print("Vértices:", g.getVertexCount())
    print("Arestas:", g.getEdgeCount())
    print("Grafo conectado?", g.isConnected())

    # exporta o grafo
    g.exportToGEPHI("grafo_teste.csv")
    print("Arquivo 'grafo_teste.csv' criado com sucesso!")


# executa se o arquivo for rodado diretamente
if __name__ == "__main__":
    gerar_grafo_exemplo()
