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
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 1rem;
        white-space: nowrap;
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Visão Geral",
    "Busca e Caminhos", 
    "Centralidade",
    "Métricas Avançadas",
    "Edição do Grafo",
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
        fig_out = px.histogram(out_degrees, nbins=10, title="Out-Degree", labels={'value': 'Out-Degree', 'count': 'Frequência'})
        st.plotly_chart(fig_out, use_container_width=True)



with tab2:
    st.header("Algoritmos de Busca e Caminhos")

    edges = st.session_state.edges
    mapping = st.session_state.mapping
    num_vertices, adjacency, in_adj, weight_map = build_graph_structures(edges, mapping)

    if mapping:
        users = sorted(mapping.values())
    else:
        users = [str(i) for i in range(num_vertices)]

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


# ========================================
# TAB 3: CENTRALIDADE
# ========================================

with tab3:
    st.header("Métricas de Centralidade")
    
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
                    centralities['Closeness'] = nx.closeness_centrality(G)
                except:
                    st.warning("Closeness não pôde ser calculado (grafo desconexo)")
                    centralities['Closeness'] = {node: 0.0 for node in G.nodes()}
                
                # 4. PageRank
                centralities['PageRank'] = nx.pagerank(G)
                
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
# TAB 6: MÉTRICAS AVANÇADAS
# ========================================

with tab4:
    st.header("Métricas Avançadas")

    col1, col2 = st.columns(2)

   # ========================================
# TAB 5: MÉTRICAS AVANÇADAS
# ========================================

with tab4:
    st.header("Métricas Avançadas")

    edges = st.session_state.edges
    mapping = st.session_state.mapping
    num_vertices, adjacency, in_adj, weight_map = build_graph_structures(edges, mapping)

    # ========================================
    # MÉTRICAS DE ESTRUTURA E COESÃO
    # ========================================
    
    st.subheader("Métricas de Estrutura e Coesão")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Densidade da Rede", use_container_width=True):
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
        if st.button("Coeficiente de Aglomeração", use_container_width=True):
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
        if st.button("Assortatividade", use_container_width=True):
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
    
    st.subheader("Métricas de Comunidade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Detecção de Comunidades (Modularidade)", use_container_width=True):
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
        if st.button("Bridging Ties (Pontes entre Comunidades)", use_container_width=True):
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
# TAB 7: EDIÇÃO DO GRAFO
# ========================================

with tab5:
    st.header("Edição do Grafo")
    
    st.info("**Importante**: Certifique-se de que o grafo está carregado antes de realizar operações de edição.")
    
    edges = st.session_state.edges
    mapping = st.session_state.mapping
    num_vertices, adjacency, in_adj, weight_map = build_graph_structures(edges, mapping)
    
    # Prepare vertex list for selects (same as in other tabs)
    if mapping:
        vertices = sorted(mapping.values())
    else:
        vertices = [str(i) for i in range(num_vertices)]
    
    # ========================================
    # 1. ADICIONAR ARESTA
    # ========================================
    st.subheader("Adicionar Aresta")
    
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        add_u_label = st.selectbox("Vértice Origem (u)", vertices, key="add_u")
    
    with col2:
        add_v_label = st.selectbox("Vértice Destino (v)", vertices, key="add_v", index=1 if len(vertices) > 1 else 0)
    
    with col3:
        add_weight = st.number_input("Peso (opcional)", min_value=0.0, value=1.0, step=0.1, key="add_weight")
    
    with col4:
        st.write("")  # spacing
        st.write("")  # spacing
        if st.button("Adicionar", use_container_width=True):
            try:
                # Convert label back to index
                if mapping:
                    add_u = [k for k, v in mapping.items() if v == add_u_label][0]
                    add_v = [k for k, v in mapping.items() if v == add_v_label][0]
                else:
                    add_u = int(add_u_label)
                    add_v = int(add_v_label)
                
                # Check if edge already exists
                r_check = api_get("/graph/edge", params={"u": int(add_u), "v": int(add_v)})
                edge_exists = r_check.json().get("exists", False)
                
                if edge_exists:
                    # Get current weight and add to it
                    r_weight = api_get("/graph/edge_weight", params={"u": int(add_u), "v": int(add_v)})
                    current_weight = r_weight.json().get("weight", 0.0)
                    new_weight = current_weight + float(add_weight)
                    
                    # Update edge weight - use urljoin and requests.post directly for query params
                    import requests
                    from urllib.parse import urljoin
                    params = {"u": int(add_u), "v": int(add_v), "weight": new_weight}
                    r = requests.post(urljoin(API_BASE, "/graph/edge_weight"), params=params)
                    r.raise_for_status()
                    
                    st.success(f"Aresta ({add_u_label} → {add_v_label}) já existia (peso {current_weight:.1f}). Peso somado: {add_weight:.1f}. Novo peso total: {new_weight:.1f}")
                else:
                    # Add new edge
                    payload = {"u": int(add_u), "v": int(add_v), "weight": float(add_weight)}
                    r = api_post("/graph/edge", json=payload)
                    r.raise_for_status()
                    
                    st.success(f"Aresta ({add_u_label} → {add_v_label}) adicionada com peso {add_weight}")
                
                # Atualizar session state (recarregar edges)
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
                st.session_state.edges = edges
                
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao adicionar aresta: {e}")
    
    st.divider()
    
    # ========================================
    # 2. REMOVER ARESTA
    # ========================================
    st.subheader("Remover Aresta")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        remove_u_label = st.selectbox("Vértice Origem (u)", vertices, key="remove_u")
    
    with col2:
        remove_v_label = st.selectbox("Vértice Destino (v)", vertices, key="remove_v", index=1 if len(vertices) > 1 else 0)
    
    with col3:
        st.write("")  # spacing
        st.write("")  # spacing
        if st.button("Remover", use_container_width=True):
            try:
                # Convert label back to index
                if mapping:
                    remove_u = [k for k, v in mapping.items() if v == remove_u_label][0]
                    remove_v = [k for k, v in mapping.items() if v == remove_v_label][0]
                else:
                    remove_u = int(remove_u_label)
                    remove_v = int(remove_v_label)
                
                # Call API to remove edge
                params = {"u": int(remove_u), "v": int(remove_v)}
                r = api_get("/graph/edge", params={"u": int(remove_u), "v": int(remove_v)})
                if not r.json().get("exists"):
                    st.warning(f"Aresta ({remove_u_label} → {remove_v_label}) não existe")
                else:
                    r = requests.delete(urljoin(API_BASE, "/graph/edge"), params=params)
                    r.raise_for_status()
                    
                    st.success(f"Aresta ({remove_u_label} → {remove_v_label}) removida")
                    
                    # Atualizar session state
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
                    st.session_state.edges = edges
                    
                    st.rerun()
            except Exception as e:
                st.error(f"Erro ao remover aresta: {e}")
    
    st.divider()
    
    # ========================================
    # 3. ALTERAR PESO DE ARESTA
    # ========================================
    st.subheader("Alterar Peso de Aresta")
    
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        weight_u_label = st.selectbox("Vértice Origem (u)", vertices, key="weight_u")
    
    with col2:
        weight_v_label = st.selectbox("Vértice Destino (v)", vertices, key="weight_v", index=1 if len(vertices) > 1 else 0)
    
    with col3:
        new_edge_weight = st.number_input("Novo Peso", min_value=0.0, value=1.0, step=0.1, key="new_edge_weight")
    
    with col4:
        st.write("")  
        if st.button("Alterar", use_container_width=True):
            try:
                # Convert label back to index
                if mapping:
                    weight_u = [k for k, v in mapping.items() if v == weight_u_label][0]
                    weight_v = [k for k, v in mapping.items() if v == weight_v_label][0]
                else:
                    weight_u = int(weight_u_label)
                    weight_v = int(weight_v_label)
                
                # Check if edge exists
                r = api_get("/graph/edge", params={"u": int(weight_u), "v": int(weight_v)})
                if not r.json().get("exists"):
                    st.warning(f"Aresta ({weight_u_label} → {weight_v_label}) não existe. Adicione-a primeiro.")
                else:
                    # Call API to set edge weight
                    params = {"u": int(weight_u), "v": int(weight_v), "weight": float(new_edge_weight)}
                    r = api_post("/graph/edge_weight", params=params)
                    r.raise_for_status()
                    
                    st.success(f"Peso da aresta ({weight_u_label} → {weight_v_label}) alterado para {new_edge_weight}")
                    
                    # Atualizar session state
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
                    st.session_state.edges = edges
                    
                    st.rerun()
            except Exception as e:
                st.error(f"Erro ao alterar peso da aresta: {e}")
    
    st.divider()
    
    # ========================================
    # 4. ALTERAR PESO DE VÉRTICE
    # ========================================
    st.subheader("Alterar Peso de Vértice")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        vertex_id_label = st.selectbox("Vértice (v)", vertices, key="vertex_id")
    
    with col2:
        new_vertex_weight = st.number_input("Novo Peso do Vértice", min_value=0.0, value=0.0, step=0.1, key="new_vertex_weight")
    
    with col3:
        st.write("")  # spacing
        st.write("")  # spacing
        if st.button("Definir Peso", use_container_width=True):
            try:
                # Convert label back to index
                if mapping:
                    vertex_id = [k for k, v in mapping.items() if v == vertex_id_label][0]
                else:
                    vertex_id = int(vertex_id_label)
                
                # Call API to set vertex weight
                params = {"v": int(vertex_id), "weight": float(new_vertex_weight)}
                r = api_post("/graph/vertex_weight", params=params)
                r.raise_for_status()
                
                st.success(f"Peso do vértice {vertex_id_label} definido como {new_vertex_weight}")
            except Exception as e:
                st.error(f"Erro ao alterar peso do vértice: {e}")
    
    st.divider()
    
    # ========================================
    # 5. CONSULTAR PESO DE VÉRTICE
    # ========================================
    st.subheader("Consultar Peso de Vértice")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query_vertex_id_label = st.selectbox("Vértice (v)", vertices, key="query_vertex_id")
    
    with col2:
        st.write("")  # spacing
        st.write("")  # spacing
        if st.button("Consultar", use_container_width=True):
            try:
                # Convert label back to index
                if mapping:
                    query_vertex_id = [k for k, v in mapping.items() if v == query_vertex_id_label][0]
                else:
                    query_vertex_id = int(query_vertex_id_label)
                
                # Call API to get vertex weight
                params = {"v": int(query_vertex_id)}
                r = api_get("/graph/vertex_weight", params=params)
                r.raise_for_status()
                
                data = r.json()
                weight = data.get("weight", 0.0)
                
                st.info(f"**Peso do vértice {query_vertex_id_label}**: {weight}")
            except Exception as e:
                st.error(f"Erro ao consultar peso do vértice: {e}")
    
    # ========================================
    # VERIFICAÇÕES RÁPIDAS (client-side)
    # ========================================
    st.divider()
    st.subheader("Verificações Rápidas")
    # Usamos apenas a API para verificar; a UI converte rótulos em índices.

    def label_to_index(label):
        if mapping:
            for k, v in mapping.items():
                if v == label:
                    try:
                        return int(k)
                    except Exception:
                        return k
            raise ValueError("Label not found")
        else:
            return int(label)

    # isDivergent: same source, different targets
    with st.expander("isDivergent (mesma origem, alvos diferentes)"):
        c1, c2 = st.columns(2)
        with c1:
            d_u1 = st.selectbox("Origem 1 (u1)", vertices, key="div_u1")
            d_v1 = st.selectbox("Destino 1 (v1)", vertices, key="div_v1")
        with c2:
            d_u2 = st.selectbox("Origem 2 (u2)", vertices, key="div_u2")
            d_v2 = st.selectbox("Destino 2 (v2)", vertices, key="div_v2")
        if st.button("Verificar Divergente", key="check_divergent"):
            try:
                u1 = label_to_index(d_u1)
                v1 = label_to_index(d_v1)
                u2 = label_to_index(d_u2)
                v2 = label_to_index(d_v2)
                params = {"u1": int(u1), "v1": int(v1), "u2": int(u2), "v2": int(v2)}
                r = api_get("/graph/is_divergent", params=params)
                r.raise_for_status()
                data = r.json()
                if data.get("is_divergent"):
                    st.success("Divergente: True")
                else:
                    st.info("Divergente: False")
            except Exception as e:
                st.error(f"Erro ao verificar divergent: {e}")

    # isConvergent: same target, different sources
    with st.expander("isConvergent (mesmo destino, origens diferentes)"):
        c1, c2 = st.columns(2)
        with c1:
            c_u1 = st.selectbox("Origem 1 (u1)", vertices, key="conv_u1")
            c_v1 = st.selectbox("Destino 1 (v1)", vertices, key="conv_v1")
        with c2:
            c_u2 = st.selectbox("Origem 2 (u2)", vertices, key="conv_u2")
            c_v2 = st.selectbox("Destino 2 (v2)", vertices, key="conv_v2")
        if st.button("Verificar Convergente", key="check_convergent"):
            try:
                u1 = label_to_index(c_u1)
                v1 = label_to_index(c_v1)
                u2 = label_to_index(c_u2)
                v2 = label_to_index(c_v2)
                params = {"u1": int(u1), "v1": int(v1), "u2": int(u2), "v2": int(v2)}
                r = api_get("/graph/is_convergent", params=params)
                r.raise_for_status()
                data = r.json()
                if data.get("is_convergent"):
                    st.success("Convergente: True")
                else:
                    st.info("Convergente: False")
            except Exception as e:
                st.error(f"Erro ao verificar convergent: {e}")

    # isIncident: whether vertex x is incident to edge (u,v)
    with st.expander("isIncident (vértice incidente a aresta)"):
        ic1, ic2 = st.columns(2)
        with ic1:
            i_u = st.selectbox("Aresta Origem (u)", vertices, key="inc_u")
            i_v = st.selectbox("Aresta Destino (v)", vertices, key="inc_v")
        with ic2:
            i_x = st.selectbox("Vértice (x)", vertices, key="inc_x")
        if st.button("Verificar Incidente", key="check_incident"):
            try:
                u = label_to_index(i_u)
                v = label_to_index(i_v)
                x = label_to_index(i_x)
                params = {"u": int(u), "v": int(v), "x": int(x)}
                r = api_get("/graph/is_incident", params=params)
                r.raise_for_status()
                data = r.json()
                if data.get("is_incident"):
                    st.success("Incidente: True")
                else:
                    st.info("Incidente: False")
            except Exception as e:
                st.error(f"Erro ao verificar incident: {e}")

    # isCompleteGraph: check if density == 1.0 (directed complete)
    with st.expander("isCompleteGraph (grafo completo)"):
        if st.button("Verificar Grafo Completo", key="check_complete"):
            try:
                num_nodes = num_vertices
                num_edges = len(edges)
                max_edges = num_nodes * (num_nodes - 1) if num_nodes > 0 else 0
                density = (num_edges / max_edges) if max_edges > 0 else 0.0
                is_complete = (max_edges > 0) and (num_edges == max_edges)
                st.metric("Densidade", f"{density:.4f}")
                if is_complete:
                    st.success("Grafo completo: Sim (todas as arestas direcionadas presentes)")
                else:
                    st.info(f"Grafo completo: Não — {num_edges} de {max_edges} arestas ({density*100:.2f}%)")
            except Exception as e:
                st.error(f"Erro ao verificar complete: {e}")


# ========================================
# TAB 8: EXPORTAÇÃO
# ========================================

with tab6:
    st.header("Exportação de Dados")

    st.subheader("Exportar para CSV (Gephi)")
    st.info("**Formato CSV compatível com Gephi** - Exporta lista de arestas com pesos para visualização no Gephi")

    gephi_filename = st.text_input("Nome do arquivo CSV", "grafo_export.csv")

    if st.button("Exportar CSV", use_container_width=True):
        try:
            # Request server to export CSV and return file bytes
            content = api_download_bytes("/graph/export", params={"filename": gephi_filename})
            st.success(f"Arquivo CSV gerado: {gephi_filename}")

            st.download_button(
                label="Baixar arquivo CSV",
                data=content,
                file_name=gephi_filename,
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Erro ao exportar: {e}")

    st.divider()

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

