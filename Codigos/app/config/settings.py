"""Configurações da aplicação.

Centraliza credenciais e permite sobrescrever via variáveis de ambiente.
"""

from dataclasses import dataclass
import os


@dataclass
class Settings:
	neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
	neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
	neo4j_password: str = os.getenv("NEO4J_PASSWORD", "strongpassword")


settings = Settings()

