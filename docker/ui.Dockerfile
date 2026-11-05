# ShopSage Streamlit UI: a thin client of the API, so it needs only these packages.
FROM python:3.12.13-slim
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir streamlit==1.66.1 httpx==0.28.1 pandas==3.1.0
WORKDIR /app
COPY app ./app
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
