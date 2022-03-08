# build:  docker build --target builder -t streamlit-base -f Dockerfile . 
# tag:    docker image tag streamlit-base:latest localhost:5000/streamlit/streamlit-base:latest 
# upload: docker image push localhost:5000/streamlit/streamlit-base:latest 
#FROM python:3-slim AS builder
#COPY packages.txt /tmp/ 
#COPY requirements.txt /tmp/
#RUN apt-get update && xargs apt-get install -y </tmp/packages.txt
#RUN pip install --upgrade pip \
# && pip install --no-cache-dir --user -r /tmp/requirements.txt


FROM python:3-slim AS app
COPY --from=localhost:5000/streamlit/streamlit-base:latest /root/.local /root/.local
COPY *.py /app/
WORKDIR app
EXPOSE 8501
ENV PATH=/root/.local/bin:$PATH
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
