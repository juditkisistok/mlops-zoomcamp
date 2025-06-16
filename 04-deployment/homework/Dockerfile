FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv

RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ["Pipfile", "Pipfile.lock", "./"]

# we don't need to create a virtual environment
RUN pip install numpy pandas scikit-learn gunicorn
RUN pipenv install --system --deploy

COPY ["starter.py", "./"]