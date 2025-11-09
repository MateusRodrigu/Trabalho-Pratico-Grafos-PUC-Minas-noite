# file: github_to_neo4j_full.py
import os
import time
import requests
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Carrega .env
load_dotenv()
GITHUB_TOKEN= 
NEO4J_URI= "bolt://localhost:7687"
NEO4J_USER= "neo4j"
NEO4J_PASSWORD= "strongpassword"
OWNER= "DrewThomasson"
REPO= "ebook2audiobook"


if not all([GITHUB_TOKEN, OWNER, REPO, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise SystemExit("Variáveis de ambiente faltando. Defina GITHUB_TOKEN, OWNER, REPO, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Sessão requests com retry
def criar_sessao_com_retry():
    session = requests.Session()
    retry_strategy = Retry(
        total=6,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

session = criar_sessao_com_retry()

def paginate(url, params=None):
    """Pega todas as páginas de uma API que usa Link headers (GitHub)."""
    params = params or {}
    results = []
    while url:
        try:
            r = session.get(url, headers=HEADERS, params=params, timeout=60, verify=True)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
            # next link
            link = r.headers.get("Link", "")
            next_url = None
            if 'rel="next"' in link:
                parts = link.split(",")
                for p in parts:
                    if 'rel="next"' in p:
                        next_url = p.split(";")[0].strip().strip("<>")
            url = next_url
            params = None
            time.sleep(0.5)  # pequeno delay
        except requests.exceptions.RequestException as e:
            print(f"[WARN] Erro ao buscar {url}: {e}. Tentando novamente em 2s...")
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
            # Neo4j 4/5 style constraint (id único)
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Issue) REQUIRE i.id IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:PullRequest) REQUIRE p.id IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Comment) REQUIRE c.id IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Review) REQUIRE r.id IS UNIQUE")

    # ---- nós básicos ----
    def save_user(self, user):
        if not user or user.get("id") is None:
            return
        with self.driver.session() as s:
            s.run(
                "MERGE (u:User {id:$id}) "
                "SET u.login=$login, u.url=$url, u.type=$type",
                {"id": user["id"], "login": user.get("login"), "url": user.get("html_url"), "type": user.get("type")}
            )

    def save_issue(self, issue):
        if not issue or issue.get("id") is None:
            return
        with self.driver.session() as s:
            s.run(
                "MERGE (i:Issue {id:$id}) "
                "SET i.number=$number, i.title=$title, i.state=$state, i.created_at=$created_at, i.closed_at=$closed_at, i.url=$url",
                issue
            )

    def save_pr(self, pr):
        if not pr or pr.get("id") is None:
            return
        with self.driver.session() as s:
            s.run(
                "MERGE (p:PullRequest {id:$id}) "
                "SET p.number=$number, p.title=$title, p.state=$state, p.merged=$merged, p.merged_at=$merged_at, p.created_at=$created_at, p.url=$url",
                pr
            )

    def save_comment(self, comment):
        if not comment or comment.get("id") is None:
            return
        with self.driver.session() as s:
            s.run(
                "MERGE (c:Comment {id:$id}) "
                "SET c.body=$body, c.created_at=$created_at, c.type=$type",
                comment
            )

    def save_review(self, review):
        if not review or review.get("id") is None:
            return
        with self.driver.session() as s:
            s.run(
                "MERGE (r:Review {id:$id}) SET r.state=$state, r.body=$body, r.submitted_at=$submitted_at",
                review
            )

    # ---- relações ----
    def link_user_opened(self, user_id, target_label, target_id):
        if user_id is None or target_id is None:
            return
        with self.driver.session() as s:
            s.run(
                f"MATCH (u:User {{id:$uid}}), (t:{target_label} {{id:$tid}}) MERGE (u)-[:OPENED]->(t)",
                {"uid": user_id, "tid": target_id}
            )

    def link_comment_on(self, comment_id, target_label, target_id):
        if comment_id is None or target_id is None:
            return
        with self.driver.session() as s:
            s.run(
                f"MATCH (c:Comment {{id:$cid}}), (t:{target_label} {{id:$tid}}) MERGE (c)-[:ON]->(t)",
                {"cid": comment_id, "tid": target_id}
            )

    def link_user_commented(self, user_id, comment_id):
        if user_id is None or comment_id is None:
            return
        with self.driver.session() as s:
            s.run(
                "MATCH (u:User {id:$uid}), (c:Comment {id:$cid}) MERGE (u)-[:COMMENTED]->(c)",
                {"uid": user_id, "cid": comment_id}
            )

    def link_user_closed(self, user_data, target_label, target_id):
        if not user_data or user_data.get("id") is None or target_id is None:
            return
        # salva o usuário e cria relação
        self.save_user(user_data)
        with self.driver.session() as s:
            s.run(
                f"MATCH (u:User {{id:$uid}}), (t:{target_label} {{id:$tid}}) MERGE (u)-[:CLOSED]->(t)",
                {"uid": user_data["id"], "tid": target_id}
            )

    def link_user_merged(self, user_data, pr_id):
        if not user_data or user_data.get("id") is None or pr_id is None:
            return
        self.save_user(user_data)
        with self.driver.session() as s:
            s.run(
                "MATCH (u:User {id:$uid}), (p:PullRequest {id:$pid}) MERGE (u)-[:MERGED]->(p)",
                {"uid": user_data["id"], "pid": pr_id}
            )

    def link_review_to_pr(self, review_id, pr_id):
        if review_id is None or pr_id is None:
            return
        with self.driver.session() as s:
            s.run(
                "MATCH (r:Review {id:$rid}), (p:PullRequest {id:$pid}) MERGE (r)-[:REVIEW_OF]->(p)",
                {"rid": review_id, "pid": pr_id}
            )

    def link_user_wrote_review(self, user_id, review_id):
        if user_id is None or review_id is None:
            return
        with self.driver.session() as s:
            s.run(
                "MATCH (u:User {id:$uid}), (r:Review {id:$rid}) MERGE (u)-[:WROTE_REVIEW]->(r)",
                {"uid": user_id, "rid": review_id}
            )

    def link_user_approved_pr(self, user_id, pr_id):
        if user_id is None or pr_id is None:
            return
        with self.driver.session() as s:
            s.run(
                "MATCH (u:User {id:$uid}), (p:PullRequest {id:$pid}) MERGE (u)-[:APPROVED]->(p)",
                {"uid": user_id, "pid": pr_id}
            )

def fetch_and_store_all():
    saver = Neo4jSaver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    saver.create_constraints()

    # 1) Issues (note: pulls appear here too)
    issues_url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues?state=all&per_page=100"
    issues = paginate(issues_url)
    print(f"Found {len(issues)} issues/prs")
    for it in tqdm(issues, desc="Issues"):
        try:
            # Distinguish PR vs Issue (GitHub's issues endpoint returns PRs as well)
            if "pull_request" in it:
                # treat as lightweight PR node here; full PR details handled later
                pr_node = {
                    "id": it["id"],
                    "number": it.get("number"),
                    "title": it.get("title"),
                    "state": it.get("state"),
                    "merged": False,
                    "merged_at": None,
                    "created_at": it.get("created_at"),
                    "url": it.get("html_url")
                }
                saver.save_pr(pr_node)
                if it.get("user"):
                    saver.save_user({"id": it["user"]["id"], "login": it["user"].get("login"), "html_url": it["user"].get("html_url"), "type": it["user"].get("type")})
                    saver.link_user_opened(it["user"]["id"], "PullRequest", it["id"])
                # if closed_by present in this object (rare on issues list), attach
                if it.get("closed_by"):
                    saver.link_user_closed(it.get("closed_by"), "PullRequest", it["id"])
            else:
                issue_node = {
                    "id": it["id"],
                    "number": it.get("number"),
                    "title": it.get("title"),
                    "state": it.get("state"),
                    "created_at": it.get("created_at"),
                    "closed_at": it.get("closed_at"),
                    "url": it.get("html_url")
                }
                saver.save_issue(issue_node)
                if it.get("user"):
                    saver.save_user({"id": it["user"]["id"], "login": it["user"].get("login"), "html_url": it["user"].get("html_url"), "type": it["user"].get("type")})
                    saver.link_user_opened(it["user"]["id"], "Issue", it["id"])

                # se existe closed_by no item, cria relação CLOSED
                if it.get("closed_by"):
                    saver.link_user_closed(it.get("closed_by"), "Issue", it["id"])
        except Exception as e:
            print(f"[ERROR] ao processar issue/pr {it.get('id')}: {e}")
            continue

    # 2) Issue comments (repo-wide)
    ic_url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues/comments?per_page=100"
    i_comments = paginate(ic_url)
    print(f"Issue comments: {len(i_comments)}")
    for c in tqdm(i_comments, desc="Issue comments"):
        try:
            # each comment has "issue_url" like .../issues/{number}
            issue_api = c.get("issue_url")
            if not issue_api:
                continue
            issue_data = session.get(issue_api, headers=HEADERS, timeout=60, verify=True).json()
            target_id = issue_data.get("id")
            author = c.get("user") or {}
            comment = {"id": c.get("id"), "body": c.get("body"), "created_at": c.get("created_at"), "type": "issue_comment"}
            saver.save_comment(comment)
            # salva autor e liga
            if author.get("id"):
                saver.save_user({"id": author["id"], "login": author.get("login"), "html_url": author.get("html_url"), "type": author.get("type")})
                saver.link_user_commented(author["id"], comment["id"])
            # liga comment -> issue
            if target_id:
                saver.link_comment_on(comment["id"], "Issue", target_id)
            time.sleep(0.2)
        except Exception as e:
            print(f"[WARN] Erro ao processar comentário de issue {c.get('id')}: {e}")
            continue

    # 3) Pull requests (full) — to get merged info, reviews, etc
    prs_url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls?state=all&per_page=100"
    prs = paginate(prs_url)
    print(f"Pull requests (listed): {len(prs)}")
    for pr in tqdm(prs, desc="Pulls"):
        try:
            pr_number = pr.get("number")
            if pr_number is None:
                continue
            pr_detail = session.get(f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{pr_number}", headers=HEADERS, timeout=60, verify=True).json()
            pr_node = {
                "id": pr_detail.get("id"),
                "number": pr_detail.get("number"),
                "title": pr_detail.get("title"),
                "state": pr_detail.get("state"),
                "merged": pr_detail.get("merged", False),
                "merged_at": pr_detail.get("merged_at"),
                "created_at": pr_detail.get("created_at"),
                "url": pr_detail.get("html_url")
            }
            saver.save_pr(pr_node)

            # autor do PR
            if pr_detail.get("user"):
                saver.save_user({"id": pr_detail["user"]["id"], "login": pr_detail["user"].get("login"), "html_url": pr_detail["user"].get("html_url"), "type": pr_detail["user"].get("type")})
                saver.link_user_opened(pr_detail["user"]["id"], "PullRequest", pr_detail["id"])

            # quem mergeou, se existe
            if pr_detail.get("merged") and pr_detail.get("merged_by"):
                saver.link_user_merged(pr_detail.get("merged_by"), pr_detail.get("id"))

            # Reviews (list)
            reviews = paginate(f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{pr_number}/reviews?per_page=100")
            for rev in reviews:
                try:
                    author = rev.get("user") or {}
                    review_node = {"id": rev.get("id"), "state": rev.get("state"), "body": rev.get("body"), "submitted_at": rev.get("submitted_at")}
                    saver.save_review(review_node)
                    # link review -> PR
                    saver.link_review_to_pr(review_node["id"], pr_detail.get("id"))
                    # link user -> review
                    if author.get("id"):
                        saver.save_user({"id": author["id"], "login": author.get("login"), "html_url": author.get("html_url"), "type": author.get("type")})
                        saver.link_user_wrote_review(author["id"], review_node["id"])
                    # se aprovou explicitamente, cria relation APPROVED entre user e PR
                    if rev.get("state") and rev.get("state").upper() == "APPROVED" and author.get("id"):
                        saver.link_user_approved_pr(author["id"], pr_detail.get("id"))
                except Exception as e:
                    print(f"[WARN] erro ao processar review {rev.get('id')} do PR {pr_number}: {e}")
                    continue
            time.sleep(0.2)
        except Exception as e:
            print(f"[ERROR] ao processar pull {pr.get('number')}: {e}")
            continue

    # 4) Pull request review comments (diff comments)
    pr_review_comments_url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/comments?per_page=100"
    pr_rc = paginate(pr_review_comments_url)
    print(f"PR review comments (diff): {len(pr_rc)}")
    for c in tqdm(pr_rc, desc="PR review comments"):
        try:
            pr_api = c.get("pull_request_url")
            if not pr_api:
                continue
            pr_data = session.get(pr_api, headers=HEADERS, timeout=60, verify=True).json()
            target_id = pr_data.get("id")
            author = c.get("user") or {}
            comment = {"id": c.get("id"), "body": c.get("body"), "created_at": c.get("created_at"), "type": "pr_review_comment"}
            saver.save_comment(comment)
            if author.get("id"):
                saver.save_user({"id": author["id"], "login": author.get("login"), "html_url": author.get("html_url"), "type": author.get("type")})
                saver.link_user_commented(author["id"], comment["id"])
            if target_id:
                saver.link_comment_on(comment["id"], "PullRequest", target_id)
            time.sleep(0.2)
        except Exception as e:
            print(f"[WARN] erro ao processar pr review comment {c.get('id')}: {e}")
            continue

    saver.close()
    print("Import completo.")

if __name__ == "__main__":
    fetch_and_store_all()