FROM python:3.11-slim

WORKDIR /app

# Instala dependências do sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Dependências Python
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Código da aplicação
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8501

CMD ["streamlit", "run", "interface/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
