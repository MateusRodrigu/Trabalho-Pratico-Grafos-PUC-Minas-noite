from fastapi.testclient import TestClient
import importlib.util
from pathlib import Path

# Load the `api.py` module from the Codigos directory (works even without package __init__)
api_path = Path(__file__).parent / 'api.py'
spec = importlib.util.spec_from_file_location('codigos_api', str(api_path))
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
import json

client = TestClient(api.app)

print('LOAD ->', client.post('/graph/load', json={'implementation':'list','num_vertices':5}).json())
print('ADD 0->1 ->', client.post('/graph/edge', json={'u':0,'v':1,'weight':1}).json())
print('ADD 1->2 ->', client.post('/graph/edge', json={'u':1,'v':2,'weight':1}).json())
print('ADD 0->2 ->', client.post('/graph/edge', json={'u':0,'v':2,'weight':1}).json())

print('INFO ->', client.get('/graph/info').json())
print('BFS ->', client.get('/graph/bfs', params={'start_index':0}).json())
print('SHORTEST ->', client.get('/graph/shortest_path', params={'source_index':0,'target_index':2}).json())

# Export responses (CSV/text) - save to files in Codigos dir
resp = client.get('/graph/export', params={'filename':'test_gephi.csv'})
if resp.status_code == 200:
    with open('test_gephi.csv','wb') as f:
        f.write(resp.content)
    print('WROTE test_gephi.csv', len(resp.content))
else:
    print('EXPORT CSV status', resp.status_code, resp.text)

resp2 = client.get('/graph/export_edges', params={'filename':'test_edges.txt'})
if resp2.status_code == 200:
    with open('test_edges.txt','wb') as f:
        f.write(resp2.content)
    print('WROTE test_edges.txt', len(resp2.content))
else:
    print('EXPORT EDGES status', resp2.status_code, resp2.text)

print('DONE')
