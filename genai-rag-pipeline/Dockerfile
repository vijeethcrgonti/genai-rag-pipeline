FROM public.ecr.aws/lambda/python:3.11

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ingestion/ ingestion/
COPY embeddings/ embeddings/
COPY vectorstore/ vectorstore/
COPY retrieval/ retrieval/
COPY generation/ generation/
COPY api/ api/
COPY config/ config/

CMD ["api.main.handler"]
