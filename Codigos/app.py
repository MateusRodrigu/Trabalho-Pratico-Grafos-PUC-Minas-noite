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

# Use HTTP API instead of directly calling services
import requests
from urllib.parse import urljoin


# ========================================
# CONFIGURAÇÃO
# ========================================

st.set_page_config(
    page_title="Analisador de Grafos GitHub",
    page_icon="",
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

# Read API base from environment or fallback to default. Avoid using `st.secrets` to
# prevent StreamlitSecretNotFoundError when no secrets.toml exists in the environment.
API_BASE = os.environ.get("API_BASE") or "http://127.0.0.1:8000"


def api_post(path: str, json=None):
    # inject implementation param automatically from session state unless caller provided one
    impl = None
    if st.session_state.get('implementation'):
        # session stores UI label or internal; accept both
        if st.session_state['implementation'] in ("Lista de Adjacência", "list"):
            impl = 'list'
        elif st.session_state['implementation'] in ("Matriz de Adjacência", "matrix"):
            impl = 'matrix'
    body = dict(json) if json else {}
    if impl and 'implementation' not in body:
        body['implementation'] = impl
    return requests.post(urljoin(API_BASE, path), json=body)


def api_get(path: str, params=None):
    impl = None
    if st.session_state.get('implementation'):
        if st.session_state['implementation'] in ("Lista de Adjacência", "list"):
            impl = 'list'
        elif st.session_state['implementation'] in ("Matriz de Adjacência", "matrix"):
            impl = 'matrix'
    qp = dict(params) if params else {}
    if impl and 'implementation' not in qp:
        qp['implementation'] = impl
    return requests.get(urljoin(API_BASE, path), params=qp)


def api_download_text(path: str, params=None):
    r = api_get(path, params=params)
    r.raise_for_status()
    return r.text


def api_download_bytes(path: str, params=None):
    r = api_get(path, params=params)
    r.raise_for_status()
    return r.content


def build_graph_structures(edges, mapping=None):
    """Return num_vertices, adjacency, in_adj, weight_map."""
    if not edges:
        # fallback: derive num from mapping if present
        if mapping:
            try:
                num = max(int(k) for k in mapping.keys()) + 1
            except Exception:
                num = len(mapping)
        else:
            return 0, {}, {}, {}
    else:
        max_idx = 0
        for u, v, _ in edges:
            if u > max_idx:
                max_idx = u
            if v > max_idx:
                max_idx = v
        num = max_idx + 1

    adjacency = {i: [] for i in range(num)}
    in_adj = {i: [] for i in range(num)}
    weight = {}
    for u, v, w in edges:
        adjacency.setdefault(u, []).append(v)
        in_adj.setdefault(v, []).append(u)
        weight[(u, v)] = w

    return num, adjacency, in_adj, weight


def idx_label(idx, mapping=None):
    if not mapping:
        return str(idx)
    # mapping keys may be strings
    if isinstance(mapping, dict):
        if str(idx) in mapping:
            return mapping[str(idx)]
        if idx in mapping:
            return mapping[idx]
    return str(idx)

# Session state
if 'graph_loaded' not in st.session_state:
    st.session_state.graph_loaded = False
if 'graph_type' not in st.session_state:
    st.session_state.graph_type = None
if 'implementation' not in st.session_state:
    st.session_state.implementation = None
if 'mapping' not in st.session_state:
    st.session_state.mapping = None
if 'edges' not in st.session_state:
    st.session_state.edges = []


# ========================================
# HEADER
# ========================================

st.markdown('<h1 class="main-header">Analisador de Rede de Colaboração GitHub</h1>', unsafe_allow_html=True)
st.markdown("**Análise completa de grafos de colaboração usando Matriz de Adjacência e Lista de Adjacência**")
st.divider()


# ========================================
# SIDEBAR - CONFIGURAÇÕES
# ========================================

with st.sidebar:
    st.header("Configurações do Grafo")
    
    # Escolha da implementação
    implementation = st.radio(
        "Implementação do Grafo",
        ["Lista de Adjacência", "Matriz de Adjacência"],
        help="Lista: melhor para grafos esparsos (GitHub). Matriz: melhor para análises matriciais."
    )
    # persist UI choice so api helpers can inject the correct parameter
    st.session_state.implementation = implementation
    
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
    if st.button("Carregar Grafo", type="primary", use_container_width=True):
        with st.spinner(f"Construindo grafo {implementation.lower()}..."):
            try:
                # Call API to load graph from DB (the app previously used repo/service to build from DB)
                payload = {
                    "implementation": "list" if implementation == "Lista de Adjacência" else "matrix",
                    "graph_type": graph_type
                }
                r = api_post("/graph/load_db", json=payload)
                r.raise_for_status()
                info = r.json()

                # Fetch general info and mapping (mapping available after load_db)
                info = api_get(urljoin(API_BASE, "/graph/info")).json()
                try:
                    mapping = api_get(urljoin(API_BASE, "/graph/mapping")).json()
                except Exception:
                    mapping = None

                # Fetch edges (edge list) to reconstruct adjacency locally for visualization and metrics
                edges_txt = api_download_text("/graph/export_edges", params={"filename": "temp_edges.txt"})
                edges = []
                for line in edges_txt.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
                        edges.append((u, v, w))

                # Save minimal session state
                st.session_state.graph_loaded = True
                st.session_state.graph_type = graph_type_display
                st.session_state.implementation = implementation
                st.session_state.mapping = mapping
                st.session_state.edges = edges

                st.success(f"Grafo carregado: {info.get('vertices')} vértices, {info.get('edges')} arestas")
            except Exception as e:
                st.error(f"Erro ao carregar grafo: {e}")
    
    st.divider()
    
    # Informações do grafo atual
    if st.session_state.graph_loaded:
        st.subheader("Grafo Atual")
        num_vertices, adjacency, in_adj, weight_map = build_graph_structures(st.session_state.edges, st.session_state.mapping)
        st.info(f"""
        **Tipo**: {st.session_state.graph_type}  
        **Implementação**: {st.session_state.implementation}  
        **Vértices**: {num_vertices}  
        **Arestas**: {len(st.session_state.edges)}
        """)

        if st.button("Limpar Grafo", use_container_width=True):
            st.session_state.graph_loaded = False
            st.session_state.graph_type = None
            st.session_state.implementation = None
            st.session_state.mapping = None
            st.session_state.edges = []
            st.rerun()


# ========================================
# VERIFICAÇÃO DE GRAFO CARREGADO
# ========================================

if not st.session_state.graph_loaded:
    st.info("**Configure e carregue um grafo na barra lateral para começar a análise**")
    st.stop()


# ========================================
# TABS PRINCIPAIS
# ========================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Visão Geral",
    "Busca e Caminhos", 
    "Centralidade",
    "Componentes",
    "Ciclos e Ordem",
    "Métricas Avançadas",
    "Exportação"
])


# ========================================
# TAB 1: VISÃO GERAL
# ========================================

with tab1:
    st.header("Visão Geral do Grafo")

    # Reconstruct local structures from edge export
    edges = st.session_state.edges
    mapping = st.session_state.mapping
    num_vertices, adjacency, in_adj, weight_map = build_graph_structures(edges, mapping)

    # Try to fetch additional info from API (is_connected, is_empty, etc.)
    try:
        info = api_get("/graph/info").json()
    except Exception:
        info = None

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Vértices", num_vertices)
    with col2:
        st.metric("Arestas", len(edges))
    with col3:
        is_connected = info.get('is_connected') if info else None
        st.metric("Conectado", "Sim" if is_connected else ("Não" if is_connected is not None else "Não disponível"))
    with col4:
        is_empty = (len(edges) == 0)
        st.metric("Vazio", "Sim" if is_empty else "Não")

    st.divider()

    # Visualização do grafo
    st.subheader("Visualização Interativa")

    if st.button("Gerar Visualização (Pyvis)"):
        with st.spinner("Renderizando grafo..."):
            # Converte para NetworkX
            G = nx.DiGraph()

            for i in range(num_vertices):
                label = idx_label(i, mapping)
                G.add_node(label)

            for u, v, w in edges:
                u_label = idx_label(u, mapping)
                v_label = idx_label(v, mapping)
                G.add_edge(u_label, v_label, weight=w)

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
    st.subheader("Distribuição de Graus")

    col1, col2 = st.columns(2)

    with col1:
        in_degrees = [len(in_adj.get(i, [])) for i in range(num_vertices)]
        fig_in = px.histogram(in_degrees, nbins=20, title="In-Degree", labels={'value': 'In-Degree', 'count': 'Frequência'})
        st.plotly_chart(fig_in, use_container_width=True)

    with col2:
        out_degrees = [len(adjacency.get(i, [])) for i in range(num_vertices)]
        fig_out = px.histogram(out_degrees, nbins=20, title="Out-Degree", labels={'value': 'Out-Degree', 'count': 'Frequência'})
        st.plotly_chart(fig_out, use_container_width=True)


# ========================================
# TAB 2: BUSCA E CAMINHOS
# ========================================

with tab2:
    st.header("Algoritmos de Busca e Caminhos")

    edges = st.session_state.edges
    mapping = st.session_state.mapping
    num_vertices, adjacency, in_adj, weight_map = build_graph_structures(edges, mapping)

    # Prepare user list for selects
    if mapping:
        users = sorted(mapping.values())
    else:
        users = [str(i) for i in range(num_vertices)]

    st.subheader("BFS - Busca em Largura")
    col1, col2 = st.columns([2, 1])

    with col1:
        start_user_bfs = st.selectbox("Usuário inicial (BFS)", users, key="bfs_start")

    with col2:
        if st.button("Executar BFS"):
            params = {"start_user": start_user_bfs} if mapping else {"start_index": int(start_user_bfs)}
            try:
                r = api_get("/graph/bfs", params=params)
                r.raise_for_status()
                distances = r.json().get('distances', {})
                # distances keys are strings
                df_distances = pd.DataFrame([
                    {"Usuário": idx_label(int(k), mapping), "Distância": v}
                    for k, v in sorted({int(k): v for k, v in distances.items()}.items(), key=lambda x: x[1])
                ])
                st.dataframe(df_distances, use_container_width=True)
                st.info(f"Alcançados: {len(distances)} vértices")
            except Exception as e:
                st.error(f"Erro BFS: {e}")

    st.divider()

    st.subheader("DFS - Busca em Profundidade")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        start_user_dfs = st.selectbox("Usuário inicial (DFS)", users, key="dfs_start")
    with col2:
        dfs_type = st.radio("Tipo DFS", ["Iterativa", "Recursiva"], key="dfs_type")
    with col3:
        if st.button("Executar DFS"):
            params = {"start_user": start_user_dfs, "mode": "iterative" if dfs_type == "Iterativa" else "recursive"} if mapping else {"start_index": int(start_user_dfs), "mode": "iterative" if dfs_type == "Iterativa" else "recursive"}
            try:
                r = api_get("/graph/dfs", params=params)
                r.raise_for_status()
                visited = r.json().get('visited', [])
                visited_users = [idx_label(int(v), mapping) for v in visited]
                st.write("**Ordem de visita:**")
                st.write(" → ".join(visited_users[:20]) + ("..." if len(visited_users) > 20 else ""))
                st.info(f"Visitados: {len(visited)} vértices")
            except Exception as e:
                st.error(f"Erro DFS: {e}")

    st.divider()

    st.subheader("Caminho Mais Curto")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        source_user = st.selectbox("Origem", users, key="path_source")
    with col2:
        target_user = st.selectbox("Destino", users, key="path_target")
    with col3:
        if st.button("Encontrar Caminho"):
            params = {"source_user": source_user, "target_user": target_user} if mapping else {"source_index": int(source_user), "target_index": int(target_user)}
            try:
                r = api_get("/graph/shortest_path", params=params)
                r.raise_for_status()
                path = r.json().get('path')
                if path:
                    path_users = [idx_label(v, mapping) for v in path]
                    st.success(f"Caminho encontrado ({len(path)} saltos)")
                    st.write(" → ".join(path_users))
                else:
                    st.warning("Não há caminho entre esses usuários")
            except Exception as e:
                st.error(f"Erro shortest_path: {e}")

    st.divider()

    st.subheader("Dijkstra (Caminho Ponderado)")
    col1, col2 = st.columns([2, 1])
    with col1:
        start_dijkstra = st.selectbox("Usuário inicial (Dijkstra)", users, key="dijkstra")
    with col2:
        if st.button("Executar Dijkstra"):
            params = {"start_user": start_dijkstra} if mapping else {"start_index": int(start_dijkstra)}
            try:
                r = api_get("/graph/dijkstra", params=params)
                r.raise_for_status()
                data = r.json()
                distances = data.get('distances', {})
                df_dijkstra = pd.DataFrame([
                    {"Usuário": idx_label(int(v), mapping), "Distância": d}
                    for v, d in sorted({int(k): v for k, v in distances.items()}.items(), key=lambda x: x[1])
                    if d != float('inf')
                ])
                st.dataframe(df_dijkstra.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Erro Dijkstra: {e}")

    st.divider()

    st.subheader("K-Hop Neighbors")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        khop_user = st.selectbox("Usuário central", users, key="khop")
    with col2:
        k = st.number_input("K (número de saltos)", min_value=1, max_value=5, value=2)
    with col3:
        if st.button("Buscar Vizinhos"):
            params = {"vertex_user": khop_user, "k": k} if mapping else {"vertex_index": int(khop_user), "k": k}
            try:
                r = api_get("/graph/khop", params=params)
                r.raise_for_status()
                neighbors = r.json().get('neighbors', [])
                neighbor_users = [idx_label(int(v), mapping) for v in neighbors]
                st.success(f"{len(neighbors)} vizinhos encontrados a {k} saltos")
                st.write(", ".join(sorted(neighbor_users)[:30]))
            except Exception as e:
                st.error(f"Erro k-hop: {e}")


# ========================================
# TAB 3: CENTRALIDADE
# ========================================

with tab3:
    st.header(" Métricas de Centralidade")
    
    edges = st.session_state.edges
    mapping = st.session_state.mapping
    num_vertices, adjacency, in_adj, weight_map = build_graph_structures(edges, mapping)
    
    if st.button("Calcular Todas as Centralidades"):
        with st.spinner("Calculando centralidades com NetworkX..."):
            try:
                # Reconstrói grafo NetworkX a partir das arestas
                G = nx.DiGraph()
                
                for i in range(num_vertices):
                    label = idx_label(i, mapping)
                    G.add_node(label)
                
                for u, v, w in edges:
                    u_label = idx_label(u, mapping)
                    v_label = idx_label(v, mapping)
                    G.add_edge(u_label, v_label, weight=w)
                
                # Calcula todas as centralidades
                centralities = {}
                
                # 1. Degree Centrality
                centralities['Degree'] = nx.degree_centrality(G)
                
                # 2. Betweenness Centrality
                centralities['Betweenness'] = nx.betweenness_centrality(G)
                
                # 3. Closeness Centrality
                try:
                    centralities['Closeness'] = nfx.closeness_centrality(G)
                except:
                    st.warning("Closeness não pôde ser calculado (grafo desconexo)")
                    centralities['Closeness'] = {node: 0.0 for node in G.nodes()}
                
                # 4. PageRank
                centralities['PageRank'] = nx.pagerank(G)
                
                # 5. Eigenvector Centrality
                try:
                    centralities['Eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
                except:
                    st.warning("Eigenvector não pôde ser calculado")
                    centralities['Eigenvector'] = {node: 0.0 for node in G.nodes()}
                
                # Salva no session state
                st.session_state.centralities = centralities
                
                st.success(" Centralidades calculadas com sucesso!")
                
            except Exception as e:
                st.error(f"Erro ao calcular centralidades: {e}")
    
    # Exibição das centralidades
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
    st.header("Análise de Componentes")

    edges = st.session_state.edges
    mapping = st.session_state.mapping

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Componentes Fortemente Conectados")

        if st.button("Detectar SCC"):
            with st.spinner("Analisando..."):
                try:
                    r = api_get("/graph/scc")
                    r.raise_for_status()
                    sccs = r.json().get('sccs', [])
                    st.success(f"{len(sccs)} componentes encontrados")

                    sizes = sorted([len(scc) for scc in sccs], reverse=True)
                    df_sccs = pd.DataFrame({
                        'Componente': range(1, len(sizes) + 1),
                        'Tamanho': sizes
                    })
                    st.dataframe(df_sccs.head(10), use_container_width=True)
                    fig = px.bar(df_sccs.head(10), x='Componente', y='Tamanho', title='Tamanho dos SCCs')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erro ao detectar SCC: {e}")

    with col2:
        st.subheader("Componentes Fracamente Conectados")

        if st.button("Detectar WCC"):
            with st.spinner("Analisando..."):
                try:
                    r = api_get("/graph/wcc")
                    r.raise_for_status()
                    wccs = r.json().get('wccs', [])
                    st.success(f"{len(wccs)} componentes encontrados")

                    sizes = sorted([len(wcc) for wcc in wccs], reverse=True)
                    df_wccs = pd.DataFrame({
                        'Componente': range(1, len(sizes) + 1),
                        'Tamanho': sizes
                    })
                    st.dataframe(df_wccs.head(10), use_container_width=True)
                    fig = px.bar(df_wccs.head(10), x='Componente', y='Tamanho', title='Tamanho dos WCCs')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erro ao detectar WCC: {e}")


# ========================================
# TAB 5: CICLOS E ORDENAÇÃO
# ========================================

with tab5:
    st.header("Detecção de Ciclos e Ordenação Topológica")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detecção de Ciclos")

        if st.button("Verificar Ciclo"):
            try:
                r = api_get("/graph/has_cycle")
                r.raise_for_status()
                has_cycle = r.json().get('has_cycle')
                if has_cycle:
                    st.error("O grafo contém ciclos")
                else:
                    st.success("O grafo é acíclico (DAG)")
            except Exception as e:
                st.error(f"Erro ao verificar ciclo: {e}")

    with col2:
        st.subheader("Ordenação Topológica")

        if st.button("Calcular Ordem"):
            try:
                r = api_get("/graph/topo_sort")
                r.raise_for_status()
                topo_sort = r.json().get('topological_sort')
                if topo_sort:
                    topo_users = [idx_label(v, st.session_state.mapping) for v in topo_sort[:50]]
                    st.success("Ordenação topológica encontrada")
                    df_topo = pd.DataFrame({
                        'Posição': range(1, len(topo_users) + 1),
                        'Usuário': topo_users
                    })
                    st.dataframe(df_topo, use_container_width=True)
                else:
                    st.error("Grafo contém ciclos - ordenação topológica impossível")
            except Exception as e:
                st.error(f"Erro topo_sort: {e}")


# ========================================
# TAB 6: MÉTRICAS AVANÇADAS
# ========================================

with tab6:
    st.header("Métricas Avançadas")

    col1, col2 = st.columns(2)

   # ========================================
# TAB 6: MÉTRICAS AVANÇADAS
# ========================================

with tab6:
    st.header(" Métricas Avançadas")

    edges = st.session_state.edges
    mapping = st.session_state.mapping
    num_vertices, adjacency, in_adj, weight_map = build_graph_structures(edges, mapping)

    # ========================================
    # MÉTRICAS DE ESTRUTURA E COESÃO
    # ========================================
    
    st.subheader(" Métricas de Estrutura e Coesão")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(" Densidade da Rede", use_container_width=True):
            with st.spinner("Calculando densidade..."):
                try:
                    # Reconstrói NetworkX
                    G = nx.DiGraph()
                    for i in range(num_vertices):
                        G.add_node(idx_label(i, mapping))
                    for u, v, w in edges:
                        G.add_edge(idx_label(u, mapping), idx_label(v, mapping), weight=w)
                    
                    # Calcula densidade
                    density = nx.density(G)
                    
                    # Informações adicionais
                    num_edges = G.number_of_edges()
                    num_nodes = G.number_of_nodes()
                    max_edges = num_nodes * (num_nodes - 1)  # grafo direcionado
                    
                    st.metric("Densidade da Rede", f"{density:.4f}")
                    st.write(f"**Arestas existentes:** {num_edges}")
                    st.write(f"**Arestas possíveis:** {max_edges}")
                    st.write(f"**Percentual:** {density * 100:.2f}%")
                    
                    st.divider()
                    st.write("**Interpretação:**")
                    if density > 0.5:
                        st.success(" **Rede altamente colaborativa** - Mais da metade das conexões possíveis existem")
                    elif density > 0.3:
                        st.info(" **Rede moderadamente colaborativa** - Boa conectividade")
                    elif density > 0.1:
                        st.warning(" **Rede com colaboração moderada** - Algumas conexões isoladas")
                    else:
                        st.error(" **Rede esparsa** - Poucas conexões, colaboração limitada")
                    
                    st.info(" **Significado:** Indica o quão colaborativa é a rede como um todo. Valores altos sugerem que os colaboradores interagem amplamente entre si.")
                    
                except Exception as e:
                    st.error(f"Erro: {e}")
    
    with col2:
        if st.button(" Coeficiente de Aglomeração", use_container_width=True):
            with st.spinner("Calculando coeficiente de aglomeração..."):
                try:
                    # Reconstrói NetworkX
                    G = nx.DiGraph()
                    for i in range(num_vertices):
                        G.add_node(idx_label(i, mapping))
                    for u, v, w in edges:
                        G.add_edge(idx_label(u, mapping), idx_label(v, mapping), weight=w)
                    
                    # Converte para não-direcionado para clustering
                    G_undirected = G.to_undirected()
                    
                    # Calcula clustering
                    avg_clustering = nx.average_clustering(G_undirected)
                    transitivity = nx.transitivity(G_undirected)
                    
                    # Clustering por nó (top 10)
                    clustering_coeffs = nx.clustering(G_undirected)
                    top_clustered = sorted(clustering_coeffs.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    st.metric("Coef. Aglomeração Médio", f"{avg_clustering:.4f}")
                    st.metric("Transitividade Global", f"{transitivity:.4f}")
                    
                    st.divider()
                    st.write("**Top 10 Colaboradores mais \"Clustered\":**")
                    df_cluster = pd.DataFrame(top_clustered, columns=['Colaborador', 'Coeficiente'])
                    st.dataframe(df_cluster, use_container_width=True)
                    
                    st.divider()
                    st.write("**Interpretação:**")
                    if avg_clustering > 0.5:
                        st.success(" **Alta tendência de formar clusters** - Grupos coesos e bem definidos")
                    elif avg_clustering > 0.3:
                        st.info(" **Moderada formação de clusters** - Alguns grupos identificáveis")
                    else:
                        st.warning(" **Baixa formação de clusters** - Colaboração mais distribuída")
                    
                    st.info(" **Significado:** Mede a tendência de colaboradores formarem pequenos grupos muito conectados (\"clusters\"). Valores altos indicam times informais bem definidos.")
                    
                except Exception as e:
                    st.error(f"Erro: {e}")
    
    with col3:
        if st.button(" Assortatividade", use_container_width=True):
            with st.spinner("Calculando assortatividade..."):
                try:
                    # Reconstrói NetworkX
                    G = nx.DiGraph()
                    for i in range(num_vertices):
                        G.add_node(idx_label(i, mapping))
                    for u, v, w in edges:
                        G.add_edge(idx_label(u, mapping), idx_label(v, mapping), weight=w)
                    
                    # Calcula assortatividade
                    try:
                        assortativity = nx.degree_assortativity_coefficient(G)
                    except:
                        assortativity = 0.0
                    
                    st.metric("Assortatividade de Grau", f"{assortativity:.4f}")
                    
                    # Distribuição de graus
                    in_degrees = dict(G.in_degree())
                    out_degrees = dict(G.out_degree())
                    
                    avg_in = sum(in_degrees.values()) / len(in_degrees)
                    avg_out = sum(out_degrees.values()) / len(out_degrees)
                    
                    st.write(f"**Grau de entrada médio:** {avg_in:.2f}")
                    st.write(f"**Grau de saída médio:** {avg_out:.2f}")
                    
                    # Gráfico de dispersão
                    edge_degrees = []
                    for u, v in G.edges():
                        edge_degrees.append({
                            'source_degree': G.degree(u),
                            'target_degree': G.degree(v)
                        })
                    
                    if edge_degrees:
                        df_assort = pd.DataFrame(edge_degrees)
                        fig = px.scatter(df_assort, x='source_degree', y='target_degree',
                                        title='Assortatividade: Grau Origem vs Destino',
                                        labels={'source_degree': 'Grau do Colaborador Origem',
                                               'target_degree': 'Grau do Colaborador Destino'},
                                        opacity=0.5)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    st.write("**Interpretação:**")
                    if assortativity > 0.3:
                        st.success(" **Rede assortativa** - Colaboradores com muitas conexões se conectam entre si (rede centralizada em hubs)")
                    elif assortativity > -0.3:
                        st.info(" **Rede neutra** - Sem padrão claro de conexão")
                    else:
                        st.warning(" **Rede disassortativa** - Colaboradores muito conectados interagem com colaboradores menos conectados (rede mais distribuída)")
                    
                    st.info("💡 **Significado:** Mostra se colaboradores com muitas conexões tendem a se conectar entre si (assortativa > 0) ou se interagem mais com colaboradores menos conectados (disassortativa < 0).")
                    
                except Exception as e:
                    st.error(f"Erro: {e}")
    
    st.divider()
    
    # ========================================
    # MÉTRICAS DE COMUNIDADE
    # ========================================
    
    st.subheader(" Métricas de Comunidade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(" Detecção de Comunidades (Modularidade)", use_container_width=True):
            with st.spinner("Detectando comunidades e calculando modularidade..."):
                try:
                    # Reconstrói NetworkX
                    G = nx.DiGraph()
                    for i in range(num_vertices):
                        G.add_node(idx_label(i, mapping))
                    for u, v, w in edges:
                        G.add_edge(idx_label(u, mapping), idx_label(v, mapping), weight=w)
                    
                    # Converte para não-direcionado
                    G_undirected = G.to_undirected()
                    
                    # Detecta comunidades usando Greedy Modularity
                    communities_gen = nx.community.greedy_modularity_communities(G_undirected)
                    communities = {}
                    for idx, comm in enumerate(communities_gen):
                        for node in comm:
                            communities[node] = idx
                    
                    num_communities = len(set(communities.values()))
                    
                    # Calcula modularidade
                    partition = list(communities_gen)
                    modularity = nx.community.modularity(G_undirected, partition)
                    
                    st.success(f" **{num_communities} comunidades detectadas**")
                    st.metric("Modularidade", f"{modularity:.4f}")
                    
                    # Distribuição de tamanhos
                    comm_sizes = {}
                    for node, comm in communities.items():
                        comm_sizes[comm] = comm_sizes.get(comm, 0) + 1
                    
                    df_comm = pd.DataFrame([
                        {
                            "Comunidade": f"C{k+1}", 
                            "Tamanho": v, 
                            "Percentual": f"{v/num_vertices*100:.1f}%"
                        }
                        for k, v in sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)
                    ])
                    
                    st.dataframe(df_comm, use_container_width=True, height=300)
                    
                    # Gráfico
                    fig = px.bar(df_comm, x='Comunidade', y='Tamanho', 
                                 title='Distribuição de Comunidades',
                                 color='Tamanho',
                                 color_continuous_scale='viridis',
                                 text='Percentual')
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    st.write("**Interpretação da Modularidade:**")
                    if modularity > 0.4:
                        st.success("🔹 **Modularidade alta** - Comunidades muito bem definidas, times informais claros")
                    elif modularity > 0.3:
                        st.info("🔹 **Modularidade boa** - Comunidades razoavelmente definidas")
                    elif modularity > 0.1:
                        st.warning("🔹 **Modularidade moderada** - Comunidades menos evidentes")
                    else:
                        st.error("🔹 **Modularidade baixa** - Estrutura de comunidades fraca")
                    
                    st.info("💡 **Significado:** Identifica grupos de colaboradores que trabalham mais frequentemente juntos (times informais dentro do projeto). A modularidade mede a qualidade dessa divisão.")
                    
                except Exception as e:
                    st.error(f"Erro ao detectar comunidades: {e}")
    
    with col2:
        if st.button(" Bridging Ties (Pontes entre Comunidades)", use_container_width=True):
            with st.spinner("Analisando pontes entre comunidades..."):
                try:
                    # Reconstrói NetworkX
                    G = nx.DiGraph()
                    for i in range(num_vertices):
                        G.add_node(idx_label(i, mapping))
                    for u, v, w in edges:
                        G.add_edge(idx_label(u, mapping), idx_label(v, mapping), weight=w)
                    
                    G_undirected = G.to_undirected()
                    
                    # Detecta comunidades primeiro
                    communities_gen = nx.community.greedy_modularity_communities(G_undirected)
                    node_to_community = {}
                    for idx, comm in enumerate(communities_gen):
                        for node in comm:
                            node_to_community[node] = idx
                    
                    # Calcula betweenness centrality (identifica pontes)
                    betweenness = nx.betweenness_centrality(G)
                    
                    # Identifica nós que conectam diferentes comunidades
                    bridge_scores = {}
                    for node in G.nodes():
                        neighbors = list(G.neighbors(node))
                        if not neighbors:
                            continue
                        
                        # Conta quantas comunidades diferentes este nó conecta
                        neighbor_communities = set()
                        for neighbor in neighbors:
                            if neighbor in node_to_community:
                                neighbor_communities.add(node_to_community[neighbor])
                        
                        # Score de ponte: número de comunidades conectadas * betweenness
                        num_connected_communities = len(neighbor_communities)
                        if num_connected_communities > 1:
                            bridge_scores[node] = {
                                'betweenness': betweenness[node],
                                'communities_connected': num_connected_communities,
                                'bridge_score': betweenness[node] * num_connected_communities
                            }
                    
                    # Top pontes
                    top_bridges = sorted(bridge_scores.items(), 
                                        key=lambda x: x[1]['bridge_score'], 
                                        reverse=True)[:15]
                    
                    if top_bridges:
                        st.success(f" **{len(bridge_scores)} colaboradores atuam como pontes**")
                        
                        df_bridges = pd.DataFrame([
                            {
                                'Colaborador': node,
                                'Comunidades Conectadas': data['communities_connected'],
                                'Betweenness': f"{data['betweenness']:.4f}",
                                'Bridge Score': f"{data['bridge_score']:.4f}"
                            }
                            for node, data in top_bridges
                        ])
                        
                        st.write("**Top 15 Colaboradores-Ponte:**")
                        st.dataframe(df_bridges, use_container_width=True, height=400)
                        
                        # Gráfico
                        fig = px.bar(df_bridges.head(10), 
                                    x='Colaborador', 
                                    y='Comunidades Conectadas',
                                    title='Top 10 Pontes - Comunidades Conectadas',
                                    color='Comunidades Conectadas',
                                    color_continuous_scale='blues')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.divider()
                        st.write("**Interpretação:**")
                        max_communities = max(d['communities_connected'] for _, d in top_bridges)
                        if max_communities >= 4:
                            st.success(" **Pontes fortes identificadas** - Alguns colaboradores conectam muitas comunidades diferentes")
                        elif max_communities >= 3:
                            st.info(" **Pontes moderadas** - Colaboradores conectam algumas comunidades")
                        else:
                            st.warning("**Pontes limitadas** - Poucas conexões entre comunidades diferentes")
                        
                        st.info(" **Significado:** Identifica colaboradores que conectam diferentes comunidades, atuando como elo entre grupos que, de outra forma, seriam isolados. Esses colaboradores são críticos para a integração do projeto.")
                    else:
                        st.warning("Nenhuma ponte significativa detectada")
                    
                except Exception as e:
                    st.error(f"Erro ao analisar pontes: {e}")
# ========================================
# TAB 7: EXPORTAÇÃO
# ========================================

with tab7:
    st.header("Exportação de Dados")

    st.subheader("Exportar para Gephi")

    gephi_filename = st.text_input("Nome do arquivo", "grafo_export.csv")

    if st.button("Exportar para Gephi"):
        try:
            # Request server to export CSV and return file bytes
            content = api_download_bytes("/graph/export", params={"filename": gephi_filename})
            st.success(f"Pedido de exportação enviado: {gephi_filename}")

            st.download_button(
                "Download CSV",
                content,
                file_name=gephi_filename,
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Erro ao exportar: {e}")

    st.divider()

    st.subheader("Exportar Lista de Arestas")

    edge_filename = st.text_input("Nome do arquivo de arestas", "edge_list.txt")

    if st.session_state.implementation == "Lista de Adjacência":
        if st.button("Exportar Lista"):
            try:
                # Download plain text edge list from API
                edges_txt = api_download_text("/graph/export_edges", params={"filename": edge_filename})
                st.success(f"Lista de arestas gerada: {edge_filename}")

                st.download_button(
                    "Download Lista de Arestas",
                    edges_txt,
                    file_name=edge_filename,
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Erro ao exportar lista: {e}")

    st.divider()

    st.subheader("Estatísticas em JSON")

    if st.button("Gerar Estatísticas"):
        import json
        try:
            info = api_get("/graph/info").json()
        except Exception:
            info = {}

        stats = {
            'vertices': info.get('vertices', 0),
            'edges': info.get('edges', 0),
            'implementation': st.session_state.implementation,
            'graph_type': st.session_state.graph_type,
            'is_connected': info.get('is_connected'),
            'is_empty': info.get('is_empty'),
            'is_complete': info.get('is_complete')
        }

        json_str = json.dumps(stats, indent=2)

        st.code(json_str, language='json')

        st.download_button(
            "Download JSON",
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
    <p>Desenvolvido com carinho usando Streamlit, NetworkX e Neo4j</p>
</div>
""", unsafe_allow_html=True)


# ========================================
# FUNÇÕES AUXILIARES ADICIONAIS
# ========================================

def show_graph_comparison():
    """Mostra comparação entre implementações."""
    st.sidebar.divider()
    st.sidebar.subheader("Comparação de Implementações")
    
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

with st.expander("Consultas Customizadas (Avançado)"):
    st.subheader("Executar Operações Personalizadas")

    edges = st.session_state.edges
    mapping = st.session_state.mapping
    num_vertices, adjacency, in_adj, weight_map = build_graph_structures(edges, mapping)

    # build user lists
    if mapping:
        users_list = sorted(mapping.values())
        user_to_index = {v: int(k) for k, v in mapping.items()}
    else:
        users_list = [str(i) for i in range(num_vertices)]
        user_to_index = {str(i): i for i in range(num_vertices)}

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

    if operation == "Verificar se dois vértices são sucessores":
        col1, col2 = st.columns(2)
        with col1:
            u_user = st.selectbox("Vértice U", users_list, key="succ_u")
        with col2:
            v_user = st.selectbox("Vértice V", users_list, key="succ_v")

        if st.button("Verificar"):
            u = user_to_index[u_user]
            v = user_to_index[v_user]
            try:
                r = api_get("/graph/is_sucessor", params={"u": u, "v": v})
                r.raise_for_status()
                result = r.json().get('is_sucessor')
                if result:
                    st.success(f"{v_user} é sucessor de {u_user}")
                else:
                    st.info(f"{v_user} NÃO é sucessor de {u_user}")
            except Exception as e:
                st.error(f"Erro: {e}")

    elif operation == "Verificar se dois vértices são predecessores":
        col1, col2 = st.columns(2)
        with col1:
            u_user = st.selectbox("Vértice U", users_list, key="pred_u")
        with col2:
            v_user = st.selectbox("Vértice V", users_list, key="pred_v")

        if st.button("Verificar"):
            u = user_to_index[u_user]
            v = user_to_index[v_user]
            try:
                r = api_get("/graph/is_predecessor", params={"u": u, "v": v})
                r.raise_for_status()
                result = r.json().get('is_predecessor')
                if result:
                    st.success(f"{u_user} é predecessor de {v_user}")
                else:
                    st.info(f"{u_user} NÃO é predecessor de {v_user}")
            except Exception as e:
                st.error(f"Erro: {e}")

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
            u1 = user_to_index[u1_user]
            v1 = user_to_index[v1_user]
            u2 = user_to_index[u2_user]
            v2 = user_to_index[v2_user]
            try:
                r = api_get("/graph/is_divergent", params={"u1": u1, "v1": v1, "u2": u2, "v2": v2})
                r.raise_for_status()
                result = r.json().get('is_divergent')
                if result:
                    st.success(f"As arestas ({u1_user}→{v1_user}) e ({u2_user}→{v2_user}) são DIVERGENTES")
                else:
                    st.info(f"As arestas NÃO são divergentes")
            except Exception as e:
                st.error(f"Erro: {e}")

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
            u1 = user_to_index[u1_user]
            v1 = user_to_index[v1_user]
            u2 = user_to_index[u2_user]
            v2 = user_to_index[v2_user]
            try:
                r = api_get("/graph/is_convergent", params={"u1": u1, "v1": v1, "u2": u2, "v2": v2})
                r.raise_for_status()
                result = r.json().get('is_convergent')
                if result:
                    st.success(f"As arestas ({u1_user}→{v1_user}) e ({u2_user}→{v2_user}) são CONVERGENTES")
                else:
                    st.info(f"As arestas NÃO são convergentes")
            except Exception as e:
                st.error(f"Erro: {e}")

    elif operation == "Verificar se vértice é incidente a aresta":
        st.write("**Aresta:**")
        col1, col2 = st.columns(2)
        with col1:
            u_user = st.selectbox("U", users_list, key="inc_u")
        with col2:
            v_user = st.selectbox("V", users_list, key="inc_v")

        x_user = st.selectbox("Vértice X", users_list, key="inc_x")

        if st.button("Verificar"):
            u = user_to_index[u_user]
            v = user_to_index[v_user]
            x = user_to_index[x_user]
            try:
                r = api_get("/graph/is_incident", params={"u": u, "v": v, "x": x})
                r.raise_for_status()
                result = r.json().get('is_incident')
                if result:
                    st.success(f"{x_user} é INCIDENTE à aresta ({u_user}→{v_user})")
                else:
                    st.info(f"{x_user} NÃO é incidente à aresta")
            except Exception as e:
                st.error(f"Erro: {e}")

    elif operation == "Obter peso de aresta específica":
        col1, col2 = st.columns(2)
        with col1:
            u_user = st.selectbox("Origem", users_list, key="weight_u")
        with col2:
            v_user = st.selectbox("Destino", users_list, key="weight_v")

        if st.button("Obter Peso"):
            u = user_to_index[u_user]
            v = user_to_index[v_user]
            w = weight_map.get((u, v))
            if w is not None:
                st.success(f"Peso da aresta ({u_user}→{v_user}): **{w}**")
            else:
                st.warning(f"Aresta ({u_user}→{v_user}) não existe")

    elif operation == "Obter peso de vértice específico":
        v_user = st.selectbox("Vértice", users_list, key="vertex_weight")

        if st.button("Obter Peso"):
            v = user_to_index[v_user]
            try:
                r = api_get("/graph/vertex_weight", params={"v": v})
                r.raise_for_status()
                weight = r.json().get('weight', 0.0)
                st.info(f"Peso do vértice {v_user}: **{weight}**")
            except Exception as e:
                st.error(f"Erro: {e}")


# ========================================
# ANÁLISE DE USUÁRIO INDIVIDUAL
# ========================================

with st.expander("Análise Detalhada de Usuário"):
    st.subheader("Perfil Completo de Colaborador")

    edges = st.session_state.edges
    mapping = st.session_state.mapping
    num_vertices, adjacency, in_adj, weight_map = build_graph_structures(edges, mapping)

    if mapping:
        users = sorted(mapping.values())
        user_to_index = {v: int(k) for k, v in mapping.items()}
    else:
        users = [str(i) for i in range(num_vertices)]
        user_to_index = {str(i): i for i in range(num_vertices)}

    selected_user = st.selectbox("Selecione um usuário", users, key="user_profile")

    if st.button("Gerar Perfil"):
        user_idx = user_to_index[selected_user]

        # Métricas básicas via API
        try:
            r_in = api_get("/graph/vertex_in_degree", params={"v": user_idx})
            r_in.raise_for_status()
            in_deg = r_in.json().get('in_degree', 0)
            
            r_out = api_get("/graph/vertex_out_degree", params={"v": user_idx})
            r_out.raise_for_status()
            out_deg = r_out.json().get('out_degree', 0)
            
            r_weight = api_get("/graph/vertex_weight", params={"v": user_idx})
            r_weight.raise_for_status()
            vertex_weight = r_weight.json().get('weight', 0.0)
            
            total_deg = in_deg + out_deg

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("In-Degree", in_deg)
            with col2:
                st.metric("Out-Degree", out_deg)
            with col3:
                st.metric("Total Degree", total_deg)
            with col4:
                st.metric("Peso Vértice", f"{vertex_weight:.2f}")
        except Exception as e:
            st.error(f"Erro ao obter métricas: {e}")
        else:
            st.divider()

            # Vizinhos
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Conexões de Saída")
                out_neighbors = [(v, weight_map.get((user_idx, v), 0.0)) for v in adjacency.get(user_idx, [])]
                if out_neighbors:
                    df_out = pd.DataFrame([
                        {"Usuário": idx_label(v, mapping), "Peso": w}
                        for v, w in sorted(out_neighbors, key=lambda x: x[1], reverse=True)[:10]
                    ])
                    st.dataframe(df_out, use_container_width=True)
                else:
                    st.info("Sem conexões de saída")

            with col2:
                st.subheader("Conexões de Entrada")
                in_neighbors = [(u, weight_map.get((u, user_idx), 0.0)) for u in in_adj.get(user_idx, [])]
                if in_neighbors:
                    df_in = pd.DataFrame([
                        {"Usuário": idx_label(u, mapping), "Peso": w}
                        for u, w in sorted(in_neighbors, key=lambda x: x[1], reverse=True)[:10]
                    ])
                    st.dataframe(df_in, use_container_width=True)
                else:
                    st.info("Sem conexões de entrada")

            # Centralidades (se calculadas)
            if 'centralities' in st.session_state:
                st.divider()
                st.subheader("Métricas de Centralidade")
                centralities = st.session_state.centralities
                centrality_data = {'Métrica': list(centralities.keys()), 'Valor': [centralities[metric].get(selected_user, 0) for metric in centralities.keys()]}
                df_cent = pd.DataFrame(centrality_data)
                fig = px.bar(df_cent, x='Métrica', y='Valor', title=f'Centralidades de {selected_user}', color='Valor', color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)


# ========================================
# COMPARAÇÃO ENTRE USUÁRIOS
# ========================================

with st.expander("Comparar Usuários"):
    st.subheader("Comparação Entre Dois Colaboradores")

    edges = st.session_state.edges
    mapping = st.session_state.mapping
    num_vertices, adjacency, in_adj, weight_map = build_graph_structures(edges, mapping)

    if mapping:
        users_list = sorted(mapping.values())
        user_to_index = {v: int(k) for k, v in mapping.items()}
    else:
        users_list = [str(i) for i in range(num_vertices)]
        user_to_index = {str(i): i for i in range(num_vertices)}

    col1, col2 = st.columns(2)

    with col1:
        user1 = st.selectbox("Usuário 1", users_list, key="compare_user1")
    with col2:
        user2 = st.selectbox("Usuário 2", users_list, key="compare_user2")

    if st.button("Comparar"):
        idx1 = user_to_index[user1]
        idx2 = user_to_index[user2]

        try:
            # Obter métricas via API
            r1_in = api_get("/graph/vertex_in_degree", params={"v": idx1})
            r1_in.raise_for_status()
            in1 = r1_in.json().get('in_degree', 0)
            
            r1_out = api_get("/graph/vertex_out_degree", params={"v": idx1})
            r1_out.raise_for_status()
            out1 = r1_out.json().get('out_degree', 0)
            
            r1_weight = api_get("/graph/vertex_weight", params={"v": idx1})
            r1_weight.raise_for_status()
            weight1 = r1_weight.json().get('weight', 0.0)
            
            r2_in = api_get("/graph/vertex_in_degree", params={"v": idx2})
            r2_in.raise_for_status()
            in2 = r2_in.json().get('in_degree', 0)
            
            r2_out = api_get("/graph/vertex_out_degree", params={"v": idx2})
            r2_out.raise_for_status()
            out2 = r2_out.json().get('out_degree', 0)
            
            r2_weight = api_get("/graph/vertex_weight", params={"v": idx2})
            r2_weight.raise_for_status()
            weight2 = r2_weight.json().get('weight', 0.0)

            metrics_data = {
                'Métrica': ['In-Degree', 'Out-Degree', 'Total Degree', 'Peso Vértice'],
                user1: [in1, out1, in1 + out1, weight1],
                user2: [in2, out2, in2 + out2, weight2]
            }
        except Exception as e:
            st.error(f"Erro ao obter métricas: {e}")
        else:
            df_compare = pd.DataFrame(metrics_data)
            st.dataframe(df_compare, use_container_width=True)

            # Gráfico comparativo
            fig = go.Figure()
            fig.add_trace(go.Bar(name=user1, x=metrics_data['Métrica'], y=metrics_data[user1], marker_color='lightblue'))
            fig.add_trace(go.Bar(name=user2, x=metrics_data['Métrica'], y=metrics_data[user2], marker_color='lightcoral'))
            fig.update_layout(title='Comparação de Métricas', barmode='group', xaxis_title='Métrica', yaxis_title='Valor')
            st.plotly_chart(fig, use_container_width=True)

            # Verifica conexão direta
            st.divider()
            st.subheader("Conexão Direta")
            if (idx1, idx2) in weight_map:
                st.success(f"{user1} → {user2} (peso: {weight_map[(idx1, idx2)]})")
            else:
                st.info(f"Sem aresta de {user1} para {user2}")

            if (idx2, idx1) in weight_map:
                st.success(f"{user2} → {user1} (peso: {weight_map[(idx2, idx1)]})")
            else:
                st.info(f"Sem aresta de {user2} para {user1}")


# ========================================
# MODO DEBUG (DESENVOLVEDOR)
# ========================================

if st.sidebar.checkbox("Modo Debug", value=False):
    st.sidebar.divider()
    st.sidebar.subheader("Debug Info")
    
    if st.session_state.graph_loaded:
        num_vertices, adjacency, in_adj, weight_map = build_graph_structures(st.session_state.edges, st.session_state.mapping)
        st.sidebar.write("**Session State:**")
        st.sidebar.json({
            'graph_type': st.session_state.graph_type,
            'implementation': st.session_state.implementation,
            'vertices': num_vertices,
            'edges': len(st.session_state.edges),
            'has_mapping': st.session_state.mapping is not None
        })