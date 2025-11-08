# file: github_to_neo4j.py
import os, time, requests
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

GITHUB_TOKEN = os.environ['GITHUB_TOKEN']
OWNER = os.environ['OWNER']
REPO = os.environ['REPO']

NEO4J_URI = os.environ['NEO4J_URI']
NEO4J_USER = os.environ['NEO4J_USER']
NEO4J_PASSWORD = os.environ['NEO4J_PASSWORD']

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Configurar sessão com retry strategy
def criar_sessao_com_retry():
    session = requests.Session()
    
    # Configurar retry strategy
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Criar sessão global
session = criar_sessao_com_retry()

def paginate(url, params=None):
    params = params or {}
    results = []
    while url:
        try:
            r = session.get(url, headers=HEADERS, params=params, timeout=30, verify=False)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
            
            # find link header for next
            link = r.headers.get('Link', '')
            next_url = None
            if 'rel="next"' in link:
                parts = link.split(',')
                for p in parts:
                    if 'rel="next"' in p:
                        next_url = p.split(';')[0].strip().strip('<>')
            url = next_url
            params = None
            
            # Adicionar pequeno delay para não sobrecarregar a API
            time.sleep(0.5)
        except requests.exceptions.RequestException as e:
            print(f"Erro ao buscar {url}: {e}")
            time.sleep(2)
            continue
    return results

class Neo4jSaver:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Issue) REQUIRE i.id IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:PullRequest) REQUIRE p.id IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Comment) REQUIRE c.id IS UNIQUE")

    def save_user(self, user):
        with self.driver.session() as s:
            s.run(
                "MERGE (u:User {id: $id}) "
                "SET u.login=$login, u.url=$url, u.type=$type",
                user
            )

    def save_issue(self, issue):
        with self.driver.session() as s:
            s.run(
                "MERGE (i:Issue {id: $id}) "
                "SET i.number=$number, i.title=$title, i.state=$state, i.created_at=$created_at, i.closed_at=$closed_at, i.url=$url",
                issue
            )

    def save_pr(self, pr):
        with self.driver.session() as s:
            s.run(
                "MERGE (p:PullRequest {id: $id}) "
                "SET p.number=$number, p.title=$title, p.state=$state, p.merged=$merged, p.merged_at=$merged_at, p.created_at=$created_at, p.url=$url",
                pr
            )

    def save_comment_on(self, comment, author, target_label, target_id):
        with self.driver.session() as s:
            # save comment
            s.run(
                "MERGE (c:Comment {id: $id}) "
                "SET c.body = $body, c.created_at=$created_at, c.type=$type",
                comment
            )
            # save author
            s.run(
                "MERGE (u:User {id: $uid}) SET u.login=$login, u.url=$url",
                {"uid": author["id"], "login": author.get("login"), "url": author.get("html_url")}
            )
            # link comment -> author
            s.run(
                "MATCH (c:Comment {id:$cid}), (u:User {id:$uid}) MERGE (u)-[:COMMENTED]->(c)",
                {"cid": comment["id"], "uid": author["id"]}
            )
            # link comment -> target (Issue or PullRequest)
            s.run(
                f"MATCH (c:Comment {{id:$cid}}), (t:{target_label} {{id:$tid}}) MERGE (c)-[:ON]->(t)",
                {"cid": comment["id"], "tid": target_id}
            )

    def link_user_opened(self, user_id, target_label, target_id):
        with self.driver.session() as s:
            s.run(
                f"MATCH (u:User {{id:$uid}}), (t:{target_label} {{id:$tid}}) MERGE (u)-[:OPENED]->(t)",
                {"uid": user_id, "tid": target_id}
            )

    def save_review(self, review, author, pr_id):
        with self.driver.session() as s:
            s.run(
                "MERGE (r:Review {id: $id}) SET r.state=$state, r.body=$body, r.submitted_at=$submitted_at",
                review
            )
            s.run(
                "MERGE (u:User {id: $uid}) SET u.login=$login, u.url=$url",
                {"uid": author["id"], "login": author.get("login"), "url": author.get("html_url")}
            )
            s.run(
                "MATCH (r:Review {id:$rid}), (u:User {id:$uid}) MERGE (u)-[:WROTE_REVIEW]->(r)",
                {"rid": review["id"], "uid": author["id"]}
            )
            s.run(
                "MATCH (r:Review {id:$rid}), (p:PullRequest {id:$pid}) MERGE (r)-[:REVIEW_OF]->(p)",
                {"rid": review["id"], "pid": pr_id}
            )
            # if approve, create relation
            if review.get("state") and review["state"].upper() == "APPROVED":
                s.run(
                    "MATCH (u:User {id:$uid}), (p:PullRequest {id:$pid}) MERGE (u)-[:APPROVED]->(p)",
                    {"uid": author["id"], "pid": pr_id}
                )

def fetch_and_store_all():
    saver = Neo4jSaver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    saver.create_constraints()

    # 1) Issues (note: pulls appear here too)
    issues_url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues?state=all&per_page=100"
    issues = paginate(issues_url)
    print(f"Found {len(issues)} issues/prs")
    for it in tqdm(issues):
        # distinguish PR vs Issue
        if "pull_request" in it:
            # basic PR info (we'll fetch full PR separately)
            pr_node = {
                "id": it["id"],
                "number": it["number"],
                "title": it.get("title"),
                "state": it.get("state"),
                "merged": False,
                "merged_at": None,
                "created_at": it.get("created_at"),
                "url": it.get("html_url")
            }
            saver.save_pr(pr_node)
            if it.get("user"):
                saver.save_user({"id": it["user"]["id"], "login": it["user"]["login"], "url": it["user"].get("html_url"), "type": it["user"].get("type")})
                saver.link_user_opened(it["user"]["id"], "PullRequest", it["id"])
        else:
            issue_node = {
                "id": it["id"],
                "number": it["number"],
                "title": it.get("title"),
                "state": it.get("state"),
                "created_at": it.get("created_at"),
                "closed_at": it.get("closed_at"),
                "url": it.get("html_url")
            }
            saver.save_issue(issue_node)
            if it.get("user"):
                saver.save_user({"id": it["user"]["id"], "login": it["user"]["login"], "url": it["user"].get("html_url"), "type": it["user"].get("type")})
                saver.link_user_opened(it["user"]["id"], "Issue", it["id"])

    # 2) Issue comments (repo-wide)
    ic_url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues/comments?per_page=100"
    i_comments = paginate(ic_url)
    print(f"Issue comments: {len(i_comments)}")
    for c in tqdm(i_comments):
        try:
            # each comment has "issue_url" like .../issues/{number}
            issue_api = c.get("issue_url")
            # determine issue id by fetching that issue? alternativ: map by issue number -> need to fetch issue to get its 'id' field
            # Simpler: fetch the issue to get its internal id
            issue_data = session.get(issue_api, headers=HEADERS, timeout=30, verify=False).json()
            target_id = issue_data["id"]
            author = c.get("user") or {"id": None, "login": None, "html_url": None}
            comment = {"id": c["id"], "body": c.get("body"), "created_at": c.get("created_at"), "type": "issue_comment"}
            saver.save_comment_on(comment, author, "Issue", target_id)
            time.sleep(0.5)
        except Exception as e:
            print(f"Erro ao processar comentário de issue {c.get('id')}: {e}")
            continue

    # 3) Pull requests (full) — to get merged info
    prs_url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls?state=all&per_page=100"
    prs = paginate(prs_url)
    print(f"Pull requests: {len(prs)}")
    for pr in tqdm(prs):
        try:
            # fetch full pr detail (to get merged, merged_at)
            pr_detail = session.get(f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{pr['number']}", headers=HEADERS, timeout=30, verify=False).json()
            pr_node = {
                "id": pr_detail["id"],
                "number": pr_detail["number"],
                "title": pr_detail.get("title"),
                "state": pr_detail.get("state"),
                "merged": pr_detail.get("merged", False),
                "merged_at": pr_detail.get("merged_at"),
                "created_at": pr_detail.get("created_at"),
                "url": pr_detail.get("html_url")
            }
            saver.save_pr(pr_node)
            if pr_detail.get("user"):
                saver.save_user({"id": pr_detail["user"]["id"], "login": pr_detail["user"]["login"], "url": pr_detail["user"].get("html_url"), "type": pr_detail["user"].get("type")})
                saver.link_user_opened(pr_detail["user"]["id"], "PullRequest", pr_detail["id"])

            # PR reviews
            reviews = paginate(f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{pr['number']}/reviews?per_page=100")
            for rev in reviews:
                author = rev.get("user") or {"id": None, "login": None, "html_url": None}
                review_node = {"id": rev["id"], "state": rev.get("state"), "body": rev.get("body"), "submitted_at": rev.get("submitted_at")}
                saver.save_review(review_node, author, pr_detail["id"])
            
            time.sleep(0.5)
        except Exception as e:
            print(f"Erro ao processar pull request {pr.get('number')}: {e}")
            continue

    # 4) Pull request review comments (diff comments)
    pr_review_comments_url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/comments?per_page=100"
    pr_rc = paginate(pr_review_comments_url)
    print(f"PR review comments (diff): {len(pr_rc)}")
    for c in tqdm(pr_rc):
        try:
            # comment has "pull_request_url"
            pr_api = c.get("pull_request_url")
            pr_data = session.get(pr_api, headers=HEADERS, timeout=30, verify=False).json()
            target_id = pr_data["id"]
            author = c.get("user") or {"id": None, "login": None, "html_url": None}
            comment = {"id": c["id"], "body": c.get("body"), "created_at": c.get("created_at"), "type": "pr_review_comment"}
            saver.save_comment_on(comment, author, "PullRequest", target_id)
            time.sleep(0.5)
        except Exception as e:
            print(f"Erro ao processar comentário de review {c.get('id')}: {e}")
            continue

    saver.close()

if __name__ == "__main__":
    fetch_and_store_all()
