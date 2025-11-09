import streamlit as st
from app.config.settings import settings
from app.services.neo4j_service import Neo4jService
from app.services.graph_service import GraphService
from app.controllers.graph_controller import GraphController
from app.utils.gephi_exporter import graph_to_html
from app.utils.graph_analyzer import summarize_graph


# ===========================
# Factory / DI com cache de recurso
# ===========================
@st.cache_resource
def get_controller() -> GraphController:
    neo4j = Neo4jService(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    graph_service = GraphService(neo4j)
    controller = GraphController(graph_service, graph_to_html)
    return controller


controller = get_controller()

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
        G, html = controller.get_graph_html(tipo)
        if len(G.nodes) == 0:
            st.warning("Nenhum dado encontrado para este tipo de relação.")
        else:
            st.success(f"Grafo com {len(G.nodes)} nós e {len(G.edges)} arestas.")
            st.components.v1.html(html, height=650, scrolling=True)

            # ===========================
            # Métricas detalhadas do grafo
            # ===========================
            metrics = summarize_graph(G)
            glob = metrics["global"]
            st.subheader("📊 Métricas Globais")
            st.markdown(
                f"""
                - Vértices: **{glob['vertex_count']}**\n
                - Arestas: **{glob['edge_count']}**\n
                - Conexo (forte): **{glob['is_connected']}**\n
                - Completo: **{glob['is_complete']}**\n
                - Vazio: **{glob['is_empty']}**\n
                - Grau médio de entrada: **{glob['average_in_degree']:.2f}**\n
                - Grau médio de saída: **{glob['average_out_degree']:.2f}**\n
                - Máx in-degree: **{glob['max_in_degree']}** | Máx out-degree: **{glob['max_out_degree']}**\n
                - Mín in-degree: **{glob['min_in_degree']}** | Mín out-degree: **{glob['min_out_degree']}**
                """
            )

            st.subheader("🧩 Vértices")
            st.dataframe(metrics["vertices"], use_container_width=True)

            st.subheader("🔗 Arestas")
            st.dataframe(metrics["edges"], use_container_width=True)
