FROM registry.access.redhat.com/ubi9/python-312:9.5

USER root

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .env .
COPY config.py .
COPY llm.py .
COPY main.py .
COPY models.py .
COPY retrieval.py .
COPY router.py .
COPY utils.py .
COPY entrypoint.sh .

RUN chown -R 1001:0 .

USER 1001

CMD ./entrypoint.sh
