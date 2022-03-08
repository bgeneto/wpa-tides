FROM python-base
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --user -r requirements.txt
COPY tides-st.py main.py
EXPOSE 8501
ENV PATH=/root/.local/bin:$PATH
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
