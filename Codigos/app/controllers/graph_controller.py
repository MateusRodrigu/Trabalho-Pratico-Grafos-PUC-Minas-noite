import os
from dotenv import load_dotenv
from app.services.graph_builder_service import GraphBuilderService

# Carrega variáveis de ambiente do .env
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def gerar_grafo_do_banco():
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        raise SystemExit("Erro: faltam variáveis de ambiente no .env (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)")

    builder = GraphBuilderService(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    graph, users = builder.build_graph_from_db()
    builder.close()

    graph.exportToGEPHI("matriz-de-adjacencia-neo4j.csv")
    print("Arquivo 'matriz-de-adjacencia-neo4j.csv' criado com sucesso!")

if __name__ == "__main__":
    gerar_grafo_do_banco()
