# ui/streamlit_app.py
"""
Interface Streamlit completa para análise de grafos GitHub.
Permite acesso a todas as funcionalidades de Matrix e List services.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import os

from app.repositories.neo4j_repository import Neo4jRepository
from app.services.adjancency_list_service import AdjacencyListService
# Matrix and analysis services disabled/commented per request
# from app.services.matrix_graph_service import MatrixGraphService
# from app.services.graph_analysis_service import GraphAnalysisService


# ========================================
# CONFIGURAÇÃO
# ========================================

st.set_page_config(
    page_title="Analisador de Grafos GitHub",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ========================================
# INICIALIZAÇÃO
# ========================================

@st.cache_resource
def init_services():
    """Inicializa serviços."""
    repo = Neo4jRepository()
    list_service = AdjacencyListService(repo)
    # Matrix implementation is commented out (MatrixGraphService import disabled)
    # matrix_service = MatrixGraphService(repo)
    matrix_service = None
    return repo, list_service, matrix_service

repo, list_service, matrix_service = init_services()

# Session state
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'graph_type' not in st.session_state:
    st.session_state.graph_type = None
if 'implementation' not in st.session_state:
    st.session_state.implementation = None
if 'analysis_service' not in st.session_state:
    st.session_state.analysis_service = None
if 'service' not in st.session_state:
    st.session_state.service = None


# ========================================
# HEADER
# ========================================

st.markdown('<h1 class="main-header">🧠 Analisador de Rede de Colaboração GitHub</h1>', unsafe_allow_html=True)
st.markdown("**Análise completa de grafos de colaboração usando Matriz de Adjacência e Lista de Adjacência**")
st.divider()


# ========================================
# SIDEBAR - CONFIGURAÇÕES
# ========================================

with st.sidebar:
    st.header("⚙️ Configurações do Grafo")
    
    # Escolha da implementação
    implementation = st.radio(
        "Implementação do Grafo",
        ["Lista de Adjacência", "Matriz de Adjacência"],
        help="Lista: melhor para grafos esparsos (GitHub). Matriz: melhor para análises matriciais."
    )
    
    # Tipo de grafo
    graph_type_map = {
        "Comentários": "comments",
        "Fechamento de Issues": "issues", 
        "Revisões/Aprovações/Merges": "reviews",
        "Integrado (Todos)": "integrated"
    }
    
    graph_type_display = st.selectbox(
        "Tipo de Interação",
        list(graph_type_map.keys())
    )
    
    graph_type = graph_type_map[graph_type_display]
    
    # Botão para carregar grafo
    if st.button("🔄 Carregar Grafo", type="primary", use_container_width=True):
        with st.spinner(f"Construindo grafo {implementation.lower()}..."):
            try:
                # Escolhe serviço baseado na implementação
                if implementation == "Lista de Adjacência":
                    service = list_service
                    if graph_type == "comments":
                        graph = service.build_comments_graph()
                    elif graph_type == "issues":
                        graph = service.build_issues_graph()
                    elif graph_type == "reviews":
                        graph = service.build_reviews_graph()
                    else:
                        graph = service.build_integrated_graph()
                else:
                    # Matrix implementation is disabled/commented out.
                    st.warning("Implementação 'Matriz de Adjacência' está desabilitada. Selecione 'Lista de Adjacência' para carregar o grafo.")
                    # Fallback: keep `service` defined to avoid crashes elsewhere but do not build via matrix.
                    service = list_service
                    graph = None
                
                # Salva no session state
                st.session_state.graph = graph
                st.session_state.graph_type = graph_type_display
                st.session_state.implementation = implementation
                st.session_state.service = service
                
                # Cria serviço de análise (comentado porque GraphAnalysisService está desabilitado)
                st.session_state.analysis_service = GraphAnalysisService(
                     graph,
                     service.index_to_user
                )
                st.session_state.analysis_service = None
                
                st.success(f"✅ Grafo carregado: {graph.getVertexCount()} vértices, {graph.getEdgeCount()} arestas")
            except Exception as e:
                st.error(f"❌ Erro ao carregar grafo: {e}")
    
    st.divider()
    
    # Informações do grafo atual
    if st.session_state.graph:
        st.subheader("📊 Grafo Atual")
        st.info(f"""
        **Tipo**: {st.session_state.graph_type}  
        **Implementação**: {st.session_state.implementation}  
        **Vértices**: {st.session_state.graph.getVertexCount()}  
        **Arestas**: {st.session_state.graph.getEdgeCount()}
        """)
        
        if st.button("🗑️ Limpar Grafo", use_container_width=True):
            st.session_state.graph = None
            st.session_state.graph_type = None
            st.session_state.implementation = None
            st.session_state.analysis_service = None
            st.rerun()


# ========================================
# VERIFICAÇÃO DE GRAFO CARREGADO
# ========================================

if not st.session_state.graph:
    st.info("👈 **Configure e carregue um grafo na barra lateral para começar a análise**")
    st.stop()


# ========================================
# TABS PRINCIPAIS
# ========================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Visão Geral",
    "🔍 Busca e Caminhos", 
    "🎯 Centralidade",
    "🌐 Componentes",
    "🔄 Ciclos e Ordem",
    "📈 Métricas Avançadas",
    "💾 Exportação"
])


# ========================================
# TAB 1: VISÃO GERAL
# ========================================

with tab1:
    st.header("📊 Visão Geral do Grafo")
    
    graph = st.session_state.graph
    service = st.session_state.service
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Vértices", graph.getVertexCount())
    with col2:
        st.metric("Arestas", graph.getEdgeCount())
    with col3:
        is_connected = graph.isConnected()
        st.metric("Conectado", "✅ Sim" if is_connected else "❌ Não")
    with col4:
        is_empty = graph.isEmptyGraph()
        st.metric("Vazio", "Sim" if is_empty else "Não")
    
    st.divider()
    
    # Visualização do grafo
    st.subheader("🎨 Visualização Interativa")
    
    if st.button("Gerar Visualização (Pyvis)"):
        with st.spinner("Renderizando grafo..."):
            # Converte para NetworkX
            G = nx.DiGraph()
            for u in range(graph.getVertexCount()):
                user_u = service.index_to_user.get(u, str(u))
                G.add_node(user_u)
                
                if st.session_state.implementation == "Lista de Adjacência":
                    for v in graph.adj_list[u]:
                        user_v = service.index_to_user.get(v, str(v))
                        weight = graph.getEdgeWeight(u, v)
                        G.add_edge(user_u, user_v, weight=weight)
                else:
                    for v in range(graph.getVertexCount()):
                        if graph.hasEdge(u, v):
                            user_v = service.index_to_user.get(v, str(v))
                            weight = graph.getEdgeWeight(u, v)
                            G.add_edge(user_u, user_v, weight=weight)
            
            # Cria visualização Pyvis
            net = Network(height="600px", width="100%", bgcolor="#222", font_color="white", directed=True)
            
            for node in G.nodes():
                in_deg = G.in_degree(node)
                out_deg = G.out_degree(node)
                net.add_node(node, label=node, size=20, title=f"{node}\nIn: {in_deg}, Out: {out_deg}")
            
            for u, v, data in G.edges(data=True):
                weight = data.get('weight', 1)
                color = "#00FF00" if weight >= 6 else "#FFA500" if weight >= 4 else "#1E90FF"
                net.add_edge(u, v, value=weight, color=color, title=f"Peso: {weight}")
            
            net.set_options("""
            {
                "physics": {
                    "enabled": true,
                    "barnesHut": {
                        "gravitationalConstant": -80000,
                        "springLength": 200
                    }
                }
            }
            """)
            
            # Salva e exibe
            net.save_graph("temp_graph.html")
            with open("temp_graph.html", "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=620, scrolling=True)
    
    st.divider()
    
    # Estatísticas de grau
    st.subheader("📉 Distribuição de Graus")
    
    col1, col2 = st.columns(2)
    
    with col1:
        in_degrees = [graph.getVertexInDegree(i) for i in range(graph.getVertexCount())]
        fig_in = px.histogram(in_degrees, nbins=20, title="In-Degree", labels={'value': 'In-Degree', 'count': 'Frequência'})
        st.plotly_chart(fig_in, use_container_width=True)
    
    with col2:
        out_degrees = [graph.getVertexOutDegree(i) for i in range(graph.getVertexCount())]
        fig_out = px.histogram(out_degrees, nbins=20, title="Out-Degree", labels={'value': 'Out-Degree', 'count': 'Frequência'})
        st.plotly_chart(fig_out, use_container_width=True)


# ========================================
# TAB 2: BUSCA E CAMINHOS
# ========================================

with tab2:
    st.header("🔍 Algoritmos de Busca e Caminhos")
    
    graph = st.session_state.graph
    service = st.session_state.service
    
    # Só disponível para Lista de Adjacência
    if st.session_state.implementation == "Lista de Adjacência":
        
        st.subheader("1️⃣ BFS - Busca em Largura")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            start_user_bfs = st.selectbox("Usuário inicial (BFS)", sorted(service.user_to_index.keys()), key="bfs_start")
        
        with col2:
            if st.button("Executar BFS"):
                start_idx = service.user_to_index[start_user_bfs]
                distances = service.bfs(graph, start_idx)
                
                df_distances = pd.DataFrame([
                    {"Usuário": service.index_to_user[v], "Distância": dist}
                    for v, dist in sorted(distances.items(), key=lambda x: x[1])
                ])
                
                st.dataframe(df_distances, use_container_width=True)
                st.info(f"✅ Alcançados: {len(distances)} vértices")
        
        st.divider()
        
        st.subheader("2️⃣ DFS - Busca em Profundidade")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            start_user_dfs = st.selectbox("Usuário inicial (DFS)", sorted(service.user_to_index.keys()), key="dfs_start")
        with col2:
            dfs_type = st.radio("Tipo DFS", ["Iterativa", "Recursiva"], key="dfs_type")
        with col3:
            if st.button("Executar DFS"):
                start_idx = service.user_to_index[start_user_dfs]
                
                if dfs_type == "Iterativa":
                    visited = service.dfs_iterative(graph, start_idx)
                else:
                    visited = service.dfs_recursive(graph, start_idx)
                
                visited_users = [service.index_to_user[v] for v in visited]
                st.write("**Ordem de visita:**")
                st.write(" → ".join(visited_users[:20]) + ("..." if len(visited_users) > 20 else ""))
                st.info(f"✅ Visitados: {len(visited)} vértices")
        
        st.divider()
        
        st.subheader("3️⃣ Caminho Mais Curto")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            source_user = st.selectbox("Origem", sorted(service.user_to_index.keys()), key="path_source")
        with col2:
            target_user = st.selectbox("Destino", sorted(service.user_to_index.keys()), key="path_target")
        with col3:
            if st.button("Encontrar Caminho"):
                source_idx = service.user_to_index[source_user]
                target_idx = service.user_to_index[target_user]
                
                path = service.find_shortest_path(graph, source_idx, target_idx)
                
                if path:
                    path_users = [service.index_to_user[v] for v in path]
                    st.success(f"✅ Caminho encontrado ({len(path)} saltos)")
                    st.write(" → ".join(path_users))
                else:
                    st.warning("❌ Não há caminho entre esses usuários")
        
        st.divider()
        
        st.subheader("4️⃣ Dijkstra (Caminho Ponderado)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            start_dijkstra = st.selectbox("Usuário inicial (Dijkstra)", sorted(service.user_to_index.keys()), key="dijkstra")
        with col2:
            if st.button("Executar Dijkstra"):
                start_idx = service.user_to_index[start_dijkstra]
                distances, predecessors = service.dijkstra(graph, start_idx)
                
                df_dijkstra = pd.DataFrame([
                    {"Usuário": service.index_to_user[v], "Distância": dist}
                    for v, dist in sorted(distances.items(), key=lambda x: x[1])
                    if dist != float('inf')
                ])
                
                st.dataframe(df_dijkstra.head(20), use_container_width=True)
        
        st.divider()
        
        st.subheader("5️⃣ K-Hop Neighbors")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            khop_user = st.selectbox("Usuário central", sorted(service.user_to_index.keys()), key="khop")
        with col2:
            k = st.number_input("K (número de saltos)", min_value=1, max_value=5, value=2)
        with col3:
            if st.button("Buscar Vizinhos"):
                user_idx = service.user_to_index[khop_user]
                neighbors = service.get_k_hop_neighbors(graph, user_idx, k)
                
                neighbor_users = [service.index_to_user[v] for v in neighbors]
                st.success(f"✅ {len(neighbors)} vizinhos encontrados a {k} saltos")
                st.write(", ".join(sorted(neighbor_users)[:30]))
    
    else:
        st.info("🔹 **Algoritmos de busca e caminhos estão disponíveis apenas para Lista de Adjacência**")
        st.markdown("""
        A implementação de **Lista de Adjacência** é otimizada para:
        - BFS e DFS (iteração eficiente sobre vizinhos)
        - Algoritmos de caminho (Dijkstra, shortest path)
        - Exploração de vizinhança
        """)


# ========================================
# TAB 3: CENTRALIDADE
# ========================================

with tab3:
    st.header("🎯 Métricas de Centralidade")
    
    analysis_service = st.session_state.analysis_service

    # Centrality calculations disabled because GraphAnalysisService import was commented out.
    if st.button("Calcular Todas as Centralidades"):
        st.info("Centralidade desabilitada: `GraphAnalysisService` foi comentado no código.")
    
    if 'centralities' in st.session_state:
        centralities = st.session_state.centralities
        
        # Tabs para cada métrica
        tabs = st.tabs(list(centralities.keys()))
        
        for idx, (metric_name, metric_data) in enumerate(centralities.items()):
            with tabs[idx]:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Top 10 em tabela
                    top_10 = sorted(metric_data.items(), key=lambda x: x[1], reverse=True)[:10]
                    df = pd.DataFrame(top_10, columns=['Usuário', 'Valor'])
                    st.dataframe(df, use_container_width=True, height=400)
                
                with col2:
                    # Gráfico de barras
                    fig = px.bar(
                        df,
                        x='Valor',
                        y='Usuário',
                        orientation='h',
                        title=f'Top 10 - {metric_name}',
                        color='Valor',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)


# ========================================
# TAB 4: COMPONENTES
# ========================================

with tab4:
    st.header("🌐 Análise de Componentes")
    
    service = st.session_state.service
    graph = st.session_state.graph
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Componentes Fortemente Conectados")
        
        if st.button("Detectar SCC"):
            with st.spinner("Analisando..."):
                if st.session_state.implementation == "Lista de Adjacência":
                    sccs = service.find_strongly_connected_components(graph)
                else:
                    sccs = service.find_strongly_connected_components(graph)
                
                st.success(f"✅ {len(sccs)} componentes encontrados")
                
                # Mostra tamanho dos componentes
                sizes = sorted([len(scc) for scc in sccs], reverse=True)
                df_sccs = pd.DataFrame({
                    'Componente': range(1, len(sizes) + 1),
                    'Tamanho': sizes
                })
                
                st.dataframe(df_sccs.head(10), use_container_width=True)
                
                fig = px.bar(df_sccs.head(10), x='Componente', y='Tamanho', title='Tamanho dos SCCs')
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Componentes Fracamente Conectados")
        
        if st.button("Detectar WCC"):
            with st.spinner("Analisando..."):
                if st.session_state.implementation == "Lista de Adjacência":
                    wccs = service.find_weakly_connected_components(graph)
                    
                    st.success(f"✅ {len(wccs)} componentes encontrados")
                    
                    sizes = sorted([len(wcc) for wcc in wccs], reverse=True)
                    df_wccs = pd.DataFrame({
                        'Componente': range(1, len(sizes) + 1),
                        'Tamanho': sizes
                    })
                    
                    st.dataframe(df_wccs.head(10), use_container_width=True)
                    
                    fig = px.bar(df_wccs.head(10), x='Componente', y='Tamanho', title='Tamanho dos WCCs')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Componentes fracos disponíveis apenas para Lista de Adjacência")


# ========================================
# TAB 5: CICLOS E ORDENAÇÃO
# ========================================

with tab5:
    st.header("🔄 Detecção de Ciclos e Ordenação Topológica")
    
    if st.session_state.implementation == "Lista de Adjacência":
        service = st.session_state.service
        graph = st.session_state.graph
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Detecção de Ciclos")
            
            if st.button("Verificar Ciclo"):
                has_cycle = service.has_cycle(graph)
                
                if has_cycle:
                    st.error("❌ O grafo contém ciclos")
                    
                    cycle = service.find_cycle(graph)
                    if cycle:
                        cycle_users = [service.index_to_user[v] for v in cycle]
                        st.write("**Ciclo encontrado:**")
                        st.write(" → ".join(cycle_users))
                else:
                    st.success("✅ O grafo é acíclico (DAG)")
        
        with col2:
            st.subheader("Ordenação Topológica")
            
            if st.button("Calcular Ordem"):
                topo_sort = service.topological_sort(graph)
                
                if topo_sort:
                    st.success("✅ Ordenação topológica encontrada")
                    topo_users = [service.index_to_user[v] for v in topo_sort[:50]]
                    
                    df_topo = pd.DataFrame({
                        'Posição': range(1, len(topo_users) + 1),
                        'Usuário': topo_users
                    })
                    st.dataframe(df_topo, use_container_width=True)
                else:
                    st.error("❌ Grafo contém ciclos - ordenação topológica impossível")
    else:
        st.info("Análise de ciclos disponível apenas para Lista de Adjacência")


# ========================================
# TAB 6: MÉTRICAS AVANÇADAS
# ========================================

with tab6:
    st.header("📈 Métricas Avançadas")
    
    analysis_service = st.session_state.analysis_service
    service = st.session_state.service
    graph = st.session_state.graph
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Métricas de Estrutura")
        
        if st.button("Calcular Métricas"):
            st.info("Métricas avançadas desabilitadas: `GraphAnalysisService` foi comentado no código.")
    
    with col2:
        st.subheader("Métricas de Distância")
        
        if st.session_state.implementation == "Lista de Adjacência":
            if st.button("Calcular Distâncias"):
                with st.spinner("Calculando..."):
                    avg_path = service.calculate_average_path_length(graph)
                    diameter = service.get_graph_diameter(graph)
                    
                    st.metric("Comprimento Médio", f"{avg_path:.2f}")
                    st.metric("Diâmetro", diameter)
        else:
            st.info("Métricas de distância disponíveis para Lista de Adjacência")
    
    st.divider()
    
    st.subheader("Análise de Comunidades")
    
    if st.button("Detectar Comunidades"):
        st.info("Detecção de comunidades desabilitada: `GraphAnalysisService` foi comentado no código.")


# ========================================
# TAB 7: EXPORTAÇÃO
# ========================================

with tab7:
    st.header("💾 Exportação de Dados")
    
    graph = st.session_state.graph
    service = st.session_state.service
    
    st.subheader("Exportar para Gephi")
    
    gephi_filename = st.text_input("Nome do arquivo", "grafo_export.csv")
    
    if st.button("Exportar para Gephi"):
        try:
            graph.exportToGEPHI(gephi_filename)
            st.success(f"✅ Grafo exportado para: {gephi_filename}")
            
            with open(gephi_filename, 'r') as f:
                st.download_button(
                    "⬇️ Download CSV",
                    f,
                    file_name=gephi_filename,
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Erro ao exportar: {e}")
    
    st.divider()
    
    st.subheader("Exportar Lista de Arestas")
    
    if st.session_state.implementation == "Lista de Adjacência":
        edge_filename = st.text_input("Nome do arquivo de arestas", "edge_list.txt")
        
        if st.button("Exportar Lista"):
            try:
                service.export_edge_list(graph, edge_filename)
                st.success(f"✅ Lista de arestas exportada para: {edge_filename}")
            except Exception as e:
                st.error(f"Erro ao exportar: {e}")
    
    st.divider()
    
    st.subheader("Estatísticas em JSON")
    
    if st.button("Gerar Estatísticas"):
        import json
        
        stats = {
            'vertices': graph.getVertexCount(),
            'edges': graph.getEdgeCount(),
            'implementation': st.session_state.implementation,
            'graph_type': st.session_state.graph_type,
            'is_connected': graph.isConnected(),
            'is_empty': graph.isEmptyGraph(),
            'is_complete': graph.isCompleteGraph()
        }
        
        json_str = json.dumps(stats, indent=2)
        
        st.code(json_str, language='json')
        
        st.download_button(
            "⬇️ Download JSON",
            json_str,
            file_name="graph_statistics.json",
            mime="application/json"
        )


# ========================================
# FOOTER
# ========================================

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Analisador de Grafos GitHub</strong> | PUC Minas - Teoria de Grafos e Computabilidade</p>
    <p>Desenvolvido com ❤️ usando Streamlit, NetworkX e Neo4j</p>
</div>
""", unsafe_allow_html=True)


# ========================================
# FUNÇÕES AUXILIARES ADICIONAIS
# ========================================

def show_graph_comparison():
    """Mostra comparação entre implementações."""
    st.sidebar.divider()
    st.sidebar.subheader("📊 Comparação de Implementações")
    
    comparison_data = {
        'Característica': [
            'Acesso a Aresta',
            'Iteração Vizinhos',
            'Memória (Grafo Esparso)',
            'Memória (Grafo Denso)',
            'Melhor Para'
        ],
        'Lista de Adjacência': [
            'O(grau médio)',
            'O(grau)',
            'Eficiente',
            'Menos eficiente',
            'Grafos esparsos'
        ],
        'Matriz de Adjacência': [
            'O(1)',
            'O(n)',
            'Menos eficiente',
            'Eficiente',
            'Grafos densos'
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.sidebar.dataframe(df_comparison, use_container_width=True)

# Chama comparação
show_graph_comparison()


# ========================================
# QUERIES CUSTOMIZADAS (BONUS)
# ========================================

with st.expander("🔬 Consultas Customizadas (Avançado)"):
    st.subheader("Executar Operações Personalizadas")
    
    operation = st.selectbox(
        "Escolha uma operação",
        [
            "Verificar se dois vértices são sucessores",
            "Verificar se dois vértices são predecessores",
            "Verificar se duas arestas são divergentes",
            "Verificar se duas arestas são convergentes",
            "Verificar se vértice é incidente a aresta",
            "Obter peso de aresta específica",
            "Obter peso de vértice específico"
        ]
    )
    
    graph = st.session_state.graph
    service = st.session_state.service
    
    users_list = sorted(service.user_to_index.keys())
    
    if operation == "Verificar se dois vértices são sucessores":
        col1, col2 = st.columns(2)
        with col1:
            u_user = st.selectbox("Vértice U", users_list, key="succ_u")
        with col2:
            v_user = st.selectbox("Vértice V", users_list, key="succ_v")
        
        if st.button("Verificar"):
            u = service.user_to_index[u_user]
            v = service.user_to_index[v_user]
            
            result = graph.isSucessor(u, v)
            
            if result:
                st.success(f"✅ {v_user} é sucessor de {u_user}")
            else:
                st.info(f"❌ {v_user} NÃO é sucessor de {u_user}")
    
    elif operation == "Verificar se dois vértices são predecessores":
        col1, col2 = st.columns(2)
        with col1:
            u_user = st.selectbox("Vértice U", users_list, key="pred_u")
        with col2:
            v_user = st.selectbox("Vértice V", users_list, key="pred_v")
        
        if st.button("Verificar"):
            u = service.user_to_index[u_user]
            v = service.user_to_index[v_user]
            
            result = graph.isPredessor(u, v)
            
            if result:
                st.success(f"✅ {u_user} é predecessor de {v_user}")
            else:
                st.info(f"❌ {u_user} NÃO é predecessor de {v_user}")
    
    elif operation == "Verificar se duas arestas são divergentes":
        st.write("**Aresta 1:**")
        col1, col2 = st.columns(2)
        with col1:
            u1_user = st.selectbox("U1", users_list, key="div_u1")
        with col2:
            v1_user = st.selectbox("V1", users_list, key="div_v1")
        
        st.write("**Aresta 2:**")
        col3, col4 = st.columns(2)
        with col3:
            u2_user = st.selectbox("U2", users_list, key="div_u2")
        with col4:
            v2_user = st.selectbox("V2", users_list, key="div_v2")
        
        if st.button("Verificar"):
            u1 = service.user_to_index[u1_user]
            v1 = service.user_to_index[v1_user]
            u2 = service.user_to_index[u2_user]
            v2 = service.user_to_index[v2_user]
            
            result = graph.isDivergent(u1, v1, u2, v2)
            
            if result:
                st.success(f"✅ As arestas ({u1_user}→{v1_user}) e ({u2_user}→{v2_user}) são DIVERGENTES")
            else:
                st.info(f"❌ As arestas NÃO são divergentes")
    
    elif operation == "Verificar se duas arestas são convergentes":
        st.write("**Aresta 1:**")
        col1, col2 = st.columns(2)
        with col1:
            u1_user = st.selectbox("U1", users_list, key="conv_u1")
        with col2:
            v1_user = st.selectbox("V1", users_list, key="conv_v1")
        
        st.write("**Aresta 2:**")
        col3, col4 = st.columns(2)
        with col3:
            u2_user = st.selectbox("U2", users_list, key="conv_u2")
        with col4:
            v2_user = st.selectbox("V2", users_list, key="conv_v2")
        
        if st.button("Verificar"):
            u1 = service.user_to_index[u1_user]
            v1 = service.user_to_index[v1_user]
            u2 = service.user_to_index[u2_user]
            v2 = service.user_to_index[v2_user]
            
            result = graph.isConvergent(u1, v1, u2, v2)
            
            if result:
                st.success(f"✅ As arestas ({u1_user}→{v1_user}) e ({u2_user}→{v2_user}) são CONVERGENTES")
            else:
                st.info(f"❌ As arestas NÃO são convergentes")
    
    elif operation == "Verificar se vértice é incidente a aresta":
        st.write("**Aresta:**")
        col1, col2 = st.columns(2)
        with col1:
            u_user = st.selectbox("U", users_list, key="inc_u")
        with col2:
            v_user = st.selectbox("V", users_list, key="inc_v")
        
        x_user = st.selectbox("Vértice X", users_list, key="inc_x")
        
        if st.button("Verificar"):
            u = service.user_to_index[u_user]
            v = service.user_to_index[v_user]
            x = service.user_to_index[x_user]
            
            result = graph.isIncident(u, v, x)
            
            if result:
                st.success(f"✅ {x_user} é INCIDENTE à aresta ({u_user}→{v_user})")
            else:
                st.info(f"❌ {x_user} NÃO é incidente à aresta")
    
    elif operation == "Obter peso de aresta específica":
        col1, col2 = st.columns(2)
        with col1:
            u_user = st.selectbox("Origem", users_list, key="weight_u")
        with col2:
            v_user = st.selectbox("Destino", users_list, key="weight_v")
        
        if st.button("Obter Peso"):
            u = service.user_to_index[u_user]
            v = service.user_to_index[v_user]
            
            if graph.hasEdge(u, v):
                weight = graph.getEdgeWeight(u, v)
                st.success(f"✅ Peso da aresta ({u_user}→{v_user}): **{weight}**")
            else:
                st.warning(f"❌ Aresta ({u_user}→{v_user}) não existe")
    
    elif operation == "Obter peso de vértice específico":
        v_user = st.selectbox("Vértice", users_list, key="vertex_weight")
        
        if st.button("Obter Peso"):
            v = service.user_to_index[v_user]
            weight = graph.getVertexWeight(v)
            st.info(f"Peso do vértice {v_user}: **{weight}**")


# ========================================
# ANÁLISE DE USUÁRIO INDIVIDUAL
# ========================================

with st.expander("👤 Análise Detalhada de Usuário"):
    st.subheader("Perfil Completo de Colaborador")
    
    graph = st.session_state.graph
    service = st.session_state.service
    
    selected_user = st.selectbox(
        "Selecione um usuário",
        sorted(service.user_to_index.keys()),
        key="user_profile"
    )
    
    if st.button("Gerar Perfil"):
        user_idx = service.user_to_index[selected_user]
        
        # Métricas básicas
        in_deg = graph.getVertexInDegree(user_idx)
        out_deg = graph.getVertexOutDegree(user_idx)
        total_deg = in_deg + out_deg
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("In-Degree", in_deg)
        with col2:
            st.metric("Out-Degree", out_deg)
        with col3:
            st.metric("Total Degree", total_deg)
        with col4:
            vertex_weight = graph.getVertexWeight(user_idx)
            st.metric("Peso Vértice", f"{vertex_weight:.2f}")
        
        st.divider()
        
        # Vizinhos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔗 Conexões de Saída")
            
            if st.session_state.implementation == "Lista de Adjacência":
                out_neighbors = service.get_out_neighbors_with_weights(graph, user_idx)
                
                if out_neighbors:
                    df_out = pd.DataFrame([
                        {
                            "Usuário": service.index_to_user[v],
                            "Peso": weight
                        }
                        for v, weight in sorted(out_neighbors, key=lambda x: x[1], reverse=True)[:10]
                    ])
                    st.dataframe(df_out, use_container_width=True)
                else:
                    st.info("Sem conexões de saída")
            else:
                # Para matriz
                out_neighbors = []
                for v in range(graph.getVertexCount()):
                    if graph.hasEdge(user_idx, v):
                        weight = graph.getEdgeWeight(user_idx, v)
                        out_neighbors.append((v, weight))
                
                if out_neighbors:
                    df_out = pd.DataFrame([
                        {
                            "Usuário": service.index_to_user[v],
                            "Peso": weight
                        }
                        for v, weight in sorted(out_neighbors, key=lambda x: x[1], reverse=True)[:10]
                    ])
                    st.dataframe(df_out, use_container_width=True)
                else:
                    st.info("Sem conexões de saída")
        
        with col2:
            st.subheader("🔗 Conexões de Entrada")
            
            if st.session_state.implementation == "Lista de Adjacência":
                in_neighbors = service.get_in_neighbors(graph, user_idx)
                
                if in_neighbors:
                    in_neighbors_with_weights = [
                        (u, graph.getEdgeWeight(u, user_idx))
                        for u in in_neighbors
                    ]
                    
                    df_in = pd.DataFrame([
                        {
                            "Usuário": service.index_to_user[u],
                            "Peso": weight
                        }
                        for u, weight in sorted(in_neighbors_with_weights, key=lambda x: x[1], reverse=True)[:10]
                    ])
                    st.dataframe(df_in, use_container_width=True)
                else:
                    st.info("Sem conexões de entrada")
            else:
                # Para matriz
                in_neighbors = []
                for u in range(graph.getVertexCount()):
                    if graph.hasEdge(u, user_idx):
                        weight = graph.getEdgeWeight(u, user_idx)
                        in_neighbors.append((u, weight))
                
                if in_neighbors:
                    df_in = pd.DataFrame([
                        {
                            "Usuário": service.index_to_user[u],
                            "Peso": weight
                        }
                        for u, weight in sorted(in_neighbors, key=lambda x: x[1], reverse=True)[:10]
                    ])
                    st.dataframe(df_in, use_container_width=True)
                else:
                    st.info("Sem conexões de entrada")
        
        # Centralidades (se calculadas)
        if 'centralities' in st.session_state:
            st.divider()
            st.subheader("📊 Métricas de Centralidade")
            
            centralities = st.session_state.centralities
            
            centrality_data = {
                'Métrica': list(centralities.keys()),
                'Valor': [centralities[metric].get(selected_user, 0) for metric in centralities.keys()]
            }
            
            df_cent = pd.DataFrame(centrality_data)
            
            fig = px.bar(
                df_cent,
                x='Métrica',
                y='Valor',
                title=f'Centralidades de {selected_user}',
                color='Valor',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)


# ========================================
# COMPARAÇÃO ENTRE USUÁRIOS
# ========================================

with st.expander("⚖️ Comparar Usuários"):
    st.subheader("Comparação Entre Dois Colaboradores")
    
    graph = st.session_state.graph
    service = st.session_state.service
    users_list = sorted(service.user_to_index.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        user1 = st.selectbox("Usuário 1", users_list, key="compare_user1")
    with col2:
        user2 = st.selectbox("Usuário 2", users_list, key="compare_user2")
    
    if st.button("Comparar"):
        idx1 = service.user_to_index[user1]
        idx2 = service.user_to_index[user2]
        
        # Métricas básicas
        metrics_data = {
            'Métrica': ['In-Degree', 'Out-Degree', 'Total Degree', 'Peso Vértice'],
            user1: [
                graph.getVertexInDegree(idx1),
                graph.getVertexOutDegree(idx1),
                graph.getVertexInDegree(idx1) + graph.getVertexOutDegree(idx1),
                graph.getVertexWeight(idx1)
            ],
            user2: [
                graph.getVertexInDegree(idx2),
                graph.getVertexOutDegree(idx2),
                graph.getVertexInDegree(idx2) + graph.getVertexOutDegree(idx2),
                graph.getVertexWeight(idx2)
            ]
        }
        
        df_compare = pd.DataFrame(metrics_data)
        st.dataframe(df_compare, use_container_width=True)
        
        # Gráfico comparativo
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name=user1,
            x=metrics_data['Métrica'],
            y=metrics_data[user1],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name=user2,
            x=metrics_data['Métrica'],
            y=metrics_data[user2],
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Comparação de Métricas',
            barmode='group',
            xaxis_title='Métrica',
            yaxis_title='Valor'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Verifica conexão direta
        st.divider()
        st.subheader("Conexão Direta")
        
        if graph.hasEdge(idx1, idx2):
            weight = graph.getEdgeWeight(idx1, idx2)
            st.success(f"✅ {user1} → {user2} (peso: {weight})")
        else:
            st.info(f"❌ Sem aresta de {user1} para {user2}")
        
        if graph.hasEdge(idx2, idx1):
            weight = graph.getEdgeWeight(idx2, idx1)
            st.success(f"✅ {user2} → {user1} (peso: {weight})")
        else:
            st.info(f"❌ Sem aresta de {user2} para {user1}")


# ========================================
# MODO DEBUG (DESENVOLVEDOR)
# ========================================

if st.sidebar.checkbox("🔧 Modo Debug", value=False):
    st.sidebar.divider()
    st.sidebar.subheader("Debug Info")
    
    if st.session_state.graph:
        st.sidebar.write("**Session State:**")
        st.sidebar.json({
            'graph_type': st.session_state.graph_type,
            'implementation': st.session_state.implementation,
            'vertices': st.session_state.graph.getVertexCount(),
            'edges': st.session_state.graph.getEdgeCount(),
            'has_analysis_service': st.session_state.analysis_service is not None
        })