FROM python:3.9
WORKDIR /app
COPY requirements.txt ./requirements.txt
COPY packages.txt ./packages.txt
RUN apt-get update && xargs apt-get install -y <packages.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
COPY . /app
ENTRYPOINT ["streamlit", "run"]
CMD ["tides-st.py"]
