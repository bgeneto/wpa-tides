FROM python:3.10-slim
COPY packages.txt /
RUN apt-get update && xargs apt-get install -y </packages.txt && apt-get clean
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --user -r requirements.txt
COPY . /app
EXPOSE 8501
ENV PATH=/root/.local/bin:$PATH
ENTRYPOINT ["streamlit", "run"]
CMD ["/app/tides-st.py"]
