# base image
FROM python:3.9-slim

# make sure the pip version is the latest one
RUN pip install -U pip
RUN pip install pipenv

# define the working directory
# where the files will be copied to
WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ["Pipfile", "Pipfile.lock", "./"]

# we don't need to create a virtual environment
RUN pipenv install --system --deploy

COPY ["predict.py", "lin_reg.bin", "./"]

# in this container, the port should be open
EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]

# build the image
# docker build -t ride-duration-prediction-service:v1 .

# run the container
# --rm: remove the container after it stops running
# -p 9696:9696: map the port 9696 of the container to the port 9696 of the host
# ride-duration-prediction-service:v1: the name of the image
# docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1