# 🎬 Movie Recommender System with AI-Powered Personalization

Welcome to the **Movie Recommender System**, a smart, AI-enhanced tool that suggests movies tailored to your taste! Built with **Python**, **FastAPI**, and **Docker**, it uses **content-based filtering** and enhances personalization through **Reinforcement Learning from AI Feedback (RLAIF)** and **DistilBERT**.

Whether you're a movie lover or just browsing for a good film, this system helps you find recommendations you’ll genuinely enjoy.

---

## 🚀 Features

- **Personalized Recommendations**  
  Learns from your movie ratings and focuses on your top-rated (5-star) favorites for more accurate suggestions.

- **AI-Powered Matching**  
  Uses **DistilBERT** to analyze movie genres and improve recommendations based on nuanced preferences.

- **Fast & Lightweight API**  
  Built with **FastAPI** for speed, deployed via **Docker** for simplicity.

- **Two Recommendation Modes**  
  - 🎯 Initial suggestions based on genre preferences  
  - 🤖 AI-enhanced suggestions with RLAIF for deeper personalization

---

## 🧠 How It Works

- **Dataset**  
  Based on the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/), preprocessed for immediate use.

- **Content-Based Filtering**  
  Calculates a genre preference profile from your ratings using a weighted average.

- **AI Personalization (RLAIF)**  
  Analyzes your 5-star movies using **DistilBERT** to better distinguish subtle genre differences (e.g., Drama vs. Thriller).

- **API-Driven**  
  FastAPI serves two endpoints to fetch tailored recommendations.

---

## 📁 Project Structure
recommender_project/
├── main.py # FastAPI application
├── movies.csv # Movie metadata
├── ratings.csv # User rating data
├── movie_feature_dict.pkl # Genre vectors for movies
├── movie_genre_embeddings.pkl # AI-generated genre embeddings
├── genre_columns.pkl # List of genres
├── Dockerfile # Docker configuration
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules


---

## 🛠 Prerequisites

- **Docker** – Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine.  
- **Git** – For cloning the repository.  
- **MovieLens 100K** – Already included in the repo (preprocessed).

---

## ⚙️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/movie-recommender.git
   cd movie-recommender

## 💡 Why Use This Project?
✅ Highly Personalized – Leverages DistilBERT + RLAIF for deeper taste profiling.

⚡ Fast & Scalable – Built with FastAPI, deployable with Docker anywhere.

📦 Lightweight – Uses python:3.9-slim and CPU-only PyTorch for minimal overhead.

🔓 Open Source – Easily customizable and extendable.
