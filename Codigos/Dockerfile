FROM python:3.11-slim

WORKDIR /app

COPY Codigos/requirements.txt /app/Codigos/requirements.txt

RUN pip install --no-cache-dir -r Codigos/requirements.txt

COPY . /app

EXPOSE 8000 8501

CMD ["bash"]
