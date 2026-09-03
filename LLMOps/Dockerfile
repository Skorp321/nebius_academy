FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    OMP_NUM_THREADS=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash grep \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    jupyterlab \
    notebook \
    ipykernel \
    numpy \
    einops \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
