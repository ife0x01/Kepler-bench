FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["sh", "-c", "pytest -q && python analyze.py --e 0.5 --steps-per-orbit 200 --orbits 50"]
