FROM python-base
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --user -r requirements.txt
COPY *.py ./
EXPOSE 8501
ENV PATH=/root/.local/bin:$PATH
ENTRYPOINT ["streamlit", "run"]
CMD ["tides-st.py"]
