FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for fast Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your package, URDFs, meshes, and app.py
COPY . .
RUN pip install --no-cache-dir -e .

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

CMD ["python", "app.py"]