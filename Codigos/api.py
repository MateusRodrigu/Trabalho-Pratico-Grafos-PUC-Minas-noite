from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from threading import Lock
import os

from fastapi.responses import FileResponse

from Codigos.app.models.adjancency_list_graph import AdjacencyListGraph
from Codigos.app.models.adjacency_matrix_graph import AdjacencyMatrixGraph
from Codigos.app.repositories.neo4j_repository import Neo4jRepository
from Codigos.app.services.adjancency_list_service import AdjacencyListService

app = FastAPI(title="Graph API", version="0.1")

# Global state (in-memory) - simple prototype
state_lock = Lock()
graph_obj = None
graph_impl = None  # 'list' or 'matrix'
repo = None
list_service = None
last_index_to_user = None


class LoadGraphIn(BaseModel):
    implementation: Optional[str] = "list"  # 'list' or 'matrix'
    num_vertices: int


class EdgeIn(BaseModel):
    u: int
    v: int
    weight: Optional[float] = 1.0


class LoadFromDbIn(BaseModel):
    implementation: Optional[str] = "list"
    graph_type: Optional[str] = "integrated"  # comments, issues, reviews, integrated


def _ensure_repo_service():
    global repo, list_service
    if repo is None or list_service is None:
        repo = Neo4jRepository()
        list_service = AdjacencyListService(repo)
    return repo, list_service


def _convert_list_to_matrix(list_graph: AdjacencyListGraph) -> AdjacencyMatrixGraph:
    m = AdjacencyMatrixGraph(list_graph.num_vertices)
    for u in range(list_graph.num_vertices):
        for v in list_graph.adj_list.get(u, []):
            m.addEdge(u, v)
            w = list_graph.getEdgeWeight(u, v)
            try:
                m.setEdgeWeight(u, v, w)
            except Exception:
                # fallback: set directly in matrix if method raises
                pass
    return m


def _convert_matrix_to_list(matrix_graph: AdjacencyMatrixGraph) -> AdjacencyListGraph:
    lg = AdjacencyListGraph(matrix_graph.num_vertices)
    for u in range(matrix_graph.num_vertices):
        for v in range(matrix_graph.num_vertices):
            try:
                if matrix_graph.hasEdge(u, v):
                    lg.addEdge(u, v)
                    w = matrix_graph.getEdgeWeight(u, v)
                    try:
                        lg.setEdgeWeight(u, v, w)
                    except Exception:
                        pass
            except Exception:
                # ignore invalid accesses
                pass
    return lg


def _ensure_graph_as_list():
    """Return an AdjacencyListGraph instance for current graph_obj (convert if needed)."""
    if graph_obj is None:
        raise HTTPException(status_code=404, detail="No graph loaded")
    if isinstance(graph_obj, AdjacencyListGraph):
        return graph_obj
    # convert matrix to list
    return _convert_matrix_to_list(graph_obj)


@app.get("/graph/bfs")
def api_bfs(start_index: Optional[int] = Query(None), start_user: Optional[str] = Query(None)):
    repo, svc = _ensure_repo_service()
    lg = _ensure_graph_as_list()
    # determine start
    if start_index is None and start_user is None:
        raise HTTPException(status_code=400, detail="Provide start_index or start_user")
    if start_user is not None:
        if start_user not in svc.user_to_index:
            raise HTTPException(status_code=404, detail="user not found in mapping")
        start_index = svc.user_to_index[start_user]
    try:
        distances = svc.bfs(lg, start_index)
        # map indices to users if available
        result = {str(k): v for k, v in distances.items()}
        return {"distances": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/dfs")
def api_dfs(start_index: Optional[int] = Query(None), start_user: Optional[str] = Query(None), mode: str = Query("iterative")):
    repo, svc = _ensure_repo_service()
    lg = _ensure_graph_as_list()
    if start_index is None and start_user is None:
        raise HTTPException(status_code=400, detail="Provide start_index or start_user")
    if start_user is not None:
        if start_user not in svc.user_to_index:
            raise HTTPException(status_code=404, detail="user not found in mapping")
        start_index = svc.user_to_index[start_user]
    try:
        if mode == "iterative":
            visited = svc.dfs_iterative(lg, start_index)
        else:
            visited = svc.dfs_recursive(lg, start_index)
        return {"visited": visited}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/shortest_path")
def api_shortest_path(source_index: Optional[int] = Query(None), target_index: Optional[int] = Query(None), source_user: Optional[str] = Query(None), target_user: Optional[str] = Query(None)):
    repo, svc = _ensure_repo_service()
    lg = _ensure_graph_as_list()
    if source_index is None and source_user is None:
        raise HTTPException(status_code=400, detail="Provide source_index or source_user")
    if target_index is None and target_user is None:
        raise HTTPException(status_code=400, detail="Provide target_index or target_user")
    if source_user is not None:
        if source_user not in svc.user_to_index:
            raise HTTPException(status_code=404, detail="source user not found in mapping")
        source_index = svc.user_to_index[source_user]
    if target_user is not None:
        if target_user not in svc.user_to_index:
            raise HTTPException(status_code=404, detail="target user not found in mapping")
        target_index = svc.user_to_index[target_user]
    try:
        path = svc.find_shortest_path(lg, source_index, target_index)
        return {"path": path}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/dijkstra")
def api_dijkstra(start_index: Optional[int] = Query(None), start_user: Optional[str] = Query(None)):
    repo, svc = _ensure_repo_service()
    lg = _ensure_graph_as_list()
    if start_index is None and start_user is None:
        raise HTTPException(status_code=400, detail="Provide start_index or start_user")
    if start_user is not None:
        if start_user not in svc.user_to_index:
            raise HTTPException(status_code=404, detail="user not found in mapping")
        start_index = svc.user_to_index[start_user]
    try:
        distances, predecessors = svc.dijkstra(lg, start_index)
        return {"distances": distances, "predecessors": predecessors}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/khop")
def api_khop(vertex_index: Optional[int] = Query(None), vertex_user: Optional[str] = Query(None), k: int = Query(...)):
    repo, svc = _ensure_repo_service()
    lg = _ensure_graph_as_list()
    if vertex_index is None and vertex_user is None:
        raise HTTPException(status_code=400, detail="Provide vertex_index or vertex_user")
    if vertex_user is not None:
        if vertex_user not in svc.user_to_index:
            raise HTTPException(status_code=404, detail="user not found in mapping")
        vertex_index = svc.user_to_index[vertex_user]
    try:
        neighbors = svc.get_k_hop_neighbors(lg, vertex_index, k)
        return {"neighbors": list(neighbors)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/scc")
def api_scc():
    repo, svc = _ensure_repo_service()
    lg = _ensure_graph_as_list()
    try:
        sccs = svc.find_strongly_connected_components(lg)
        return {"sccs": [list(s) for s in sccs]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/wcc")
def api_wcc():
    repo, svc = _ensure_repo_service()
    lg = _ensure_graph_as_list()
    try:
        wccs = svc.find_weakly_connected_components(lg)
        return {"wccs": [list(s) for s in wccs]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/has_cycle")
def api_has_cycle():
    repo, svc = _ensure_repo_service()
    lg = _ensure_graph_as_list()
    try:
        result = svc.has_cycle(lg)
        return {"has_cycle": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/topo_sort")
def api_topo_sort():
    repo, svc = _ensure_repo_service()
    lg = _ensure_graph_as_list()
    try:
        topo = svc.topological_sort(lg)
        return {"topological_sort": topo}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/export_edges")
def api_export_edges(filename: Optional[str] = Query("edge_list.txt")):
    repo, svc = _ensure_repo_service()
    lg = _ensure_graph_as_list()
    safe_name = os.path.basename(filename)
    try:
        svc.export_edge_list(lg, safe_name)
        return FileResponse(path=safe_name, filename=safe_name, media_type='text/plain')
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/graph/load")
def load_graph(payload: LoadGraphIn):
    global graph_obj, graph_impl
    if payload.num_vertices <= 0:
        raise HTTPException(status_code=400, detail="num_vertices must be > 0")

    with state_lock:
        if payload.implementation and payload.implementation.lower().startswith("mat"):
            graph_obj = AdjacencyMatrixGraph(payload.num_vertices)
            graph_impl = "matrix"
        else:
            graph_obj = AdjacencyListGraph(payload.num_vertices)
            graph_impl = "list"
    return {"status": "loaded", "implementation": graph_impl, "num_vertices": payload.num_vertices}



@app.post("/graph/load_db")
def load_graph_from_db(payload: LoadFromDbIn):
    """Load graph from Neo4j using AdjacencyListService.
    payload.graph_type: comments|issues|reviews|integrated
    payload.implementation: list|matrix
    """
    global graph_obj, graph_impl, last_index_to_user
    _ensure_repo_service()

    gtype = (payload.graph_type or "integrated").lower()
    try:
        if gtype == "comments":
            lg = list_service.build_comments_graph()
        elif gtype == "issues":
            lg = list_service.build_issues_graph()
        elif gtype == "reviews":
            lg = list_service.build_reviews_graph()
        else:
            lg = list_service.build_integrated_graph()

        with state_lock:
            if payload.implementation and payload.implementation.lower().startswith("mat"):
                graph_obj = _convert_list_to_matrix(lg)
                graph_impl = "matrix"
            else:
                graph_obj = lg
                graph_impl = "list"
            # store mapping (useful for clients)
            last_index_to_user = dict(list_service.index_to_user)

        return {"status": "loaded_from_db", "implementation": graph_impl, "vertices": graph_obj.getVertexCount(), "edges": graph_obj.getEdgeCount()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/mapping")
def get_last_mapping():
    """Return last index->user mapping (if any)."""
    if last_index_to_user is None:
        raise HTTPException(status_code=404, detail="No mapping available")
    return last_index_to_user


@app.get("/graph/info")
def graph_info():
    if graph_obj is None:
        raise HTTPException(status_code=404, detail="No graph loaded")
    return {
        "implementation": graph_impl,
        "vertices": graph_obj.getVertexCount(),
        "edges": graph_obj.getEdgeCount(),
        "is_connected": graph_obj.isConnected(),
        "is_empty": graph_obj.isEmptyGraph(),
        "is_complete": graph_obj.isCompleteGraph(),
    }


@app.post("/graph/edge")
def add_edge(e: EdgeIn):
    if graph_obj is None:
        raise HTTPException(status_code=404, detail="No graph loaded")
    try:
        with state_lock:
            # addEdge should be idempotent; call it first
            graph_obj.addEdge(e.u, e.v)
            # set weight (classes may raise if edge nonexistent)
            try:
                graph_obj.setEdgeWeight(e.u, e.v, float(e.weight))
            except Exception:
                # for some implementations setEdgeWeight requires existing edge; ensure existence
                pass
        return {"status": "ok", "u": e.u, "v": e.v}
    except IndexError as ie:
        raise HTTPException(status_code=400, detail=str(ie))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/graph/edge")
def delete_edge(u: int = Query(...), v: int = Query(...)):
    if graph_obj is None:
        raise HTTPException(status_code=404, detail="No graph loaded")
    try:
        with state_lock:
            graph_obj.removeEdge(u, v)
        return {"status": "deleted", "u": u, "v": v}
    except IndexError as ie:
        raise HTTPException(status_code=400, detail=str(ie))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/edge")
def has_edge(u: int = Query(...), v: int = Query(...)):
    if graph_obj is None:
        raise HTTPException(status_code=404, detail="No graph loaded")
    try:
        exists = graph_obj.hasEdge(u, v)
        weight = graph_obj.getEdgeWeight(u, v) if exists else None
        return {"exists": exists, "weight": weight}
    except IndexError as ie:
        raise HTTPException(status_code=400, detail=str(ie))


@app.get("/graph/export")
def export_graph(filename: Optional[str] = Query("graph_export.csv")):
    if graph_obj is None:
        raise HTTPException(status_code=404, detail="No graph loaded")
    # ensure safe filename
    safe_name = os.path.basename(filename)
    try:
        with state_lock:
            graph_obj.exportToGEPHI(safe_name)
        return FileResponse(path=safe_name, filename=safe_name, media_type='text/csv')
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
