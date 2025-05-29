# Use a lightweight base image of Python
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY main.py .
COPY movies.csv .
COPY ratings.csv .
COPY movie_feature_dict.pkl .
COPY movie_genre_embeddings.pkl .
COPY genre_columns.pkl .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
