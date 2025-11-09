import streamlit as st
from neo4j import GraphDatabase
import networkx as nx
from pyvis.network import Network
import tempfile, os

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "strongpassword"
@st.cache_resource
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

driver = get_driver()

# ===========================
# Função: Buscar grafo no Neo4j
# ===========================
def buscar_grafo(tipo_relacao: str):
    with driver.session() as session:
        if tipo_relacao == "Comentários":
            query = """
            MATCH (a:User)-[:COMMENTED]->(:Comment)-[:ON]->(t)<-[:OPENED]-(b:User)
            RETURN a.login AS source, b.login AS target, 'COMENTOU' AS rel, 2 AS peso
            """

        elif tipo_relacao == "Fechamento de Issue":
            query = """
            MATCH (a:User)-[:CLOSED|:CLOSED_BY]->(i:Issue)<-[:OPENED]-(b:User)
            RETURN a.login AS source, b.login AS target, 'FECHOU' AS rel, 3 AS peso
            """

        elif tipo_relacao == "Revisões/Aprovações/Merges":
            query = """
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
            """

        elif tipo_relacao == "Integrado":
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
            """
        else:
            return nx.DiGraph()

        results = session.run(query)

        # ===========================
        # Montagem do grafo no NetworkX
        # ===========================
        G = nx.DiGraph()
        for record in results:
            a = record["source"]
            b = record["target"]

            # Captura o peso, com fallback
            peso = record.get("peso_total", record.get("peso", 1))
            rel = record.get("rel", tipo_relacao)

            if a and b:
                G.add_node(a)
                G.add_node(b)
                G.add_edge(a, b, label=rel, peso=peso, weight=peso)

        return G

# ===========================
# Função: Exibir grafo com Pyvis
# ===========================
def exibir_grafo(G: nx.Graph):
    net = Network(height="650px", width="100%", bgcolor="#111", font_color="white", directed=True)
    
    # Escala visual baseada no peso
    for node in G.nodes():
        net.add_node(node, label=node)

    for u, v, data in G.edges(data=True):
        peso = data.get("peso", 1)
        cor = "#00FF00" if peso >= 6 else "#FFA500" if peso >= 4 else "#1E90FF"
        grossura = max(1, min(peso, 10))  # espessura da aresta

        net.add_edge(
            u, v,
            label=f"{data.get('label', '')} ({peso})",
            value=peso,
            color=cor,
            width=grossura
        )

    net.set_options("""
    var options = {
      "nodes": {
        "shape": "dot",
        "size": 20
      },
      "edges": {
        "color": {"inherit": false},
        "smooth": false
      },
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
    """)
    net.save_graph("grafo.html")

    with open("grafo.html", "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=650, scrolling=True)

# ===========================
# Interface Streamlit
# ===========================
st.set_page_config(page_title="Analisador de Grafos GitHub", layout="wide")
st.title("🧠 Analisador de Rede de Colaboração - GitHub/Neo4j")

tipo = st.selectbox(
    "Escolha o tipo de grafo:",
    ["Comentários", "Fechamento de Issue", "Revisões/Aprovações/Merges", "Integrado"]
)

if st.button("🔄 Gerar Grafo"):
    with st.spinner(f"Carregando grafo '{tipo}' do Neo4j..."):
        G = buscar_grafo(tipo)
        if len(G.nodes) == 0:
            st.warning("Nenhum dado encontrado para este tipo de relação.")
        else:
            st.success(f"Grafo com {len(G.nodes)} nós e {len(G.edges)} arestas.")
            exibir_grafo(G)