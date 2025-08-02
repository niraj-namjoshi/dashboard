from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import re
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math
from kneed import KneeLocator
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import uvicorn
import logging
import openai
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
load_dotenv()

# Get API key from environment
api_k = os.getenv("api_k")
client = genai.Client(api_key=api_k)
# Initialize FastAPI app
app = FastAPI(
    title="Review Sentiment Analysis & Clustering API",
    description="API for analyzing sentiment and clustering customer reviews",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global models (loaded once at startup)
sentiment_tokenizer = None
sentiment_model = None
embedding_model = None

# Pydantic models for request/response
class ReviewRequest(BaseModel):
    raw_reviews: List[str]
    max_clusters: Optional[int] = 20
    top_representatives: Optional[int] = 6

class SentimentResult(BaseModel):
    text: str
    sentiment: int  # 0=negative, 1=neutral, 2=positive
    scores: Dict[str, float]

class ClusterResult(BaseModel):
    cluster_id: int
    size: int
    representatives: List[str]

class AnalysisResponse(BaseModel):
    total_reviews: int
    sentiment_distribution: Dict[str, int]
    positive_reviews: List[str]
    negative_neutral_reviews: List[str]
    positive_clusters: List[ClusterResult]
    negative_neutral_clusters: List[ClusterResult]
    sentiment_details: List[SentimentResult]

# Startup event to load models
@app.on_event("startup")
async def load_models():
    global sentiment_tokenizer, sentiment_model, embedding_model
    
    try:
        # Load sentiment analysis model
        MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
        sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        
        # Load embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logging.info("Models loaded successfully")
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise e

# Clean and normalize texts
def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)  # remove HTML tags
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text

# Get sentiment scores using RoBERTa
def get_sentiment_scores(texts: List[str]) -> List[Dict]:
    results = []
    
    for text in texts:
        # Tokenize and prepare for model
        encoded_text = sentiment_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        # Get model output
        with torch.no_grad():
            output = sentiment_model(**encoded_text)
        
        # Get scores and apply softmax
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        # Get sentiment label (0=negative, 1=neutral, 2=positive)
        sentiment_label = np.argmax(scores)
        results.append({
            'text': text,
            'sentiment': int(sentiment_label),
            'scores': {
                'negative': float(scores[0]),
                'neutral': float(scores[1]),
                'positive': float(scores[2])
            }
        })
    
    return results

# Separate texts by sentiment
def separate_by_sentiment(sentiment_results: List[Dict]) -> tuple:
    positive_texts = []
    negative_neutral_texts = []
    
    for result in sentiment_results:
        if result['sentiment'] == 2:  # Positive
            positive_texts.append(result['text'])
        else:  # Negative or Neutral
            negative_neutral_texts.append(result['text'])
    
    return positive_texts, negative_neutral_texts

# Get embeddings
def get_embeddings(texts: List[str]) -> np.ndarray:
    return embedding_model.encode(texts, show_progress_bar=False)

# Find optimal number of clusters
def find_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 20) -> tuple:
    n_samples = embeddings.shape[0]
    
    # Set sensible bounds
    min_clusters = max(2, min(5, n_samples // 10))
    max_clusters = min(max_clusters, n_samples // 2)
    
    if max_clusters <= min_clusters:
        return min_clusters, {}
    
    # Prepare to store results
    range_clusters = range(min_clusters, max_clusters + 1)
    inertias = []
    silhouette_scores = []
    
    # Calculate inertia and silhouette score for different cluster numbers
    for k in range_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        inertias.append(kmeans.inertia_)
        
        # Only calculate silhouette if we have enough samples
        if n_samples > k+1:
            s_score = silhouette_score(embeddings, labels)
            silhouette_scores.append(s_score)
        else:
            silhouette_scores.append(0)
    
    # Method 1: Silhouette Score (higher is better)
    silhouette_optimal = range_clusters[np.argmax(silhouette_scores)]
    
    # Method 2: Elbow Method using KneeLocator
    try:
        kneedle = KneeLocator(
            range_clusters, inertias, 
            curve='convex', direction='decreasing'
        )
        elbow_optimal = kneedle.elbow
    except:
        elbow_optimal = range_clusters[len(range_clusters) // 2]
    
    # Method 3: Square root heuristic
    sqrt_optimal = max(min_clusters, round(math.sqrt(n_samples / 2)))
    
    # Combine methods with some logic
    if elbow_optimal is not None:
        optimal_k = round((elbow_optimal + silhouette_optimal) / 2)
    else:
        optimal_k = round((silhouette_optimal + sqrt_optimal) / 2)
    
    # Ensure result is within bounds
    optimal_k = max(min_clusters, min(optimal_k, max_clusters))
    
    return optimal_k, {}

# Cluster embeddings with optimized number of clusters
def cluster_embeddings(embeddings: np.ndarray, max_clusters: int = 20) -> tuple:
    n_samples = embeddings.shape[0]
    
    if n_samples <= 1:
        return np.zeros(n_samples, dtype=int), np.array([np.mean(embeddings, axis=0)])
    
    # Find optimal number of clusters
    n_clusters, _ = find_optimal_clusters(embeddings, max_clusters)
    
    # Ensure n_clusters is valid
    n_clusters = min(n_clusters, n_samples)
    n_clusters = max(1, n_clusters)
    
    # Perform clustering with optimized number
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    return labels, kmeans.cluster_centers_

# Extract representative sentences
def top_representatives(texts: List[str], embeddings: np.ndarray, labels: np.ndarray, 
                      centroids: np.ndarray, top_n: int = 6) -> Dict:
    reps = {}
    for cluster_id in set(labels):
        idxs = np.where(labels == cluster_id)[0]
        cluster_embs = embeddings[idxs]
        centroid = centroids[cluster_id]
        
        # cosine similarity
        sims = cluster_embs.dot(centroid) / (
            np.linalg.norm(cluster_embs, axis=1) * np.linalg.norm(centroid)
        )
        top_idxs = idxs[np.argsort(sims)[-min(top_n, len(idxs)):]]
        reps[cluster_id] = [texts[i] for i in top_idxs]
    
    return reps

# Main analysis endpoint
@app.post("/analyze-reviews", response_model=AnalysisResponse)
async def analyze_reviews(request: ReviewRequest):
    try:
        # Validate input
        if not request.raw_reviews:
            raise HTTPException(status_code=400, detail="No reviews provided")
        
        if len(request.raw_reviews) > 10000:
            raise HTTPException(status_code=400, detail="Too many reviews. Maximum 10,000 allowed.")
        
        # Clean texts
        cleaned = [clean_text(r) for r in request.raw_reviews]
        
        # Get sentiment scores
        sentiment_results = get_sentiment_scores(cleaned)
        
        # Separate by sentiment
        positive_reviews, negative_neutral_reviews = separate_by_sentiment(sentiment_results)
        
        # Initialize cluster results
        positive_clusters = []
        negative_neutral_clusters = []
        
        # Cluster positive reviews if any exist
        if positive_reviews:
            pos_embs = get_embeddings(positive_reviews)
            pos_labels, pos_cents = cluster_embeddings(pos_embs, request.max_clusters)
            pos_reps = top_representatives(positive_reviews, pos_embs, pos_labels, pos_cents, request.top_representatives)
            
            for cluster_id, representatives in pos_reps.items():
                positive_clusters.append(ClusterResult(
                    cluster_id=cluster_id,
                    size=len(np.where(pos_labels == cluster_id)[0]),
                    representatives=representatives
                ))
        
        # Cluster negative/neutral reviews
        if negative_neutral_reviews:
            neg_embs = get_embeddings(negative_neutral_reviews)
            neg_labels, neg_cents = cluster_embeddings(neg_embs, request.max_clusters)
            neg_reps = top_representatives(negative_neutral_reviews, neg_embs, neg_labels, neg_cents, request.top_representatives)
            
            for cluster_id, representatives in neg_reps.items():
                negative_neutral_clusters.append(ClusterResult(
                    cluster_id=cluster_id,
                    size=len(np.where(neg_labels == cluster_id)[0]),
                    representatives=representatives
                ))
        
        # Calculate sentiment distribution
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        for result in sentiment_results:
            if result['sentiment'] == 2:
                sentiment_counts["positive"] += 1
            elif result['sentiment'] == 1:
                sentiment_counts["neutral"] += 1
            else:
                sentiment_counts["negative"] += 1
        
        # Prepare response
        response = AnalysisResponse(
            total_reviews=len(cleaned),
            sentiment_distribution=sentiment_counts,
            positive_reviews=positive_reviews,
            negative_neutral_reviews=negative_neutral_reviews,
            positive_clusters=positive_clusters,
            negative_neutral_clusters=negative_neutral_clusters,
            sentiment_details=[SentimentResult(**result) for result in sentiment_results]
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": sentiment_model is not None and embedding_model is not None}

# Root endpoint - serve the HTML file directly
@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = static_dir / "index.html"
    with open(html_file, "r") as f:
        content = f.read()
    return content
@app.post("/sentiment-chart")
async def sentiment_chart(request: ReviewRequest):
    try:
        if not request.raw_reviews:
            raise HTTPException(status_code=400, detail="No reviews provided")

        cleaned = [clean_text(r) for r in request.raw_reviews]
        sentiment_results = get_sentiment_scores(cleaned)

        # Count sentiments
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        for result in sentiment_results:
            if result['sentiment'] == 2:
                sentiment_counts["positive"] += 1
            elif result['sentiment'] == 1:
                sentiment_counts["neutral"] += 1
            else:
                sentiment_counts["negative"] += 1

        return {
            "labels": ["Positive", "Negative"],
            "data": [
                sentiment_counts["positive"],
                sentiment_counts["negative"] + sentiment_counts["neutral"]  # combine for simplicity
            ],
            "colors": ["#1e7145", "#b91d47"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment chart failed: {str(e)}")

@app.get("/chart-data")
async def get_chart_data():
    # Replace this with real analysis in production
    return JSONResponse({
        "labels": ["Positive", "Negative"],
        "data": [68, 32],  # Example data
        "colors": ["#36A2EB", "#FF6384"]
    })
# @app.post("/generate-problem")
# async def generate_problem(request: ReviewRequest):
#     try:
#         if not request.raw_reviews:
#             raise HTTPException(status_code=400, detail="No reviews provided")

#         # Clean and score
#         cleaned = [clean_text(r) for r in request.raw_reviews]
#         sentiment_results = get_sentiment_scores(cleaned)

#         # Separate by sentiment
#         positive_reviews, negative_neutral_reviews = separate_by_sentiment(sentiment_results)

#         # Choose a group to generate problem from (let’s say negative/neutral)
#         reps_text = []
#         if negative_neutral_reviews:
#             neg_embs = get_embeddings(negative_neutral_reviews)
#             neg_labels, neg_cents = cluster_embeddings(neg_embs, request.max_clusters)
#             neg_reps = top_representatives(
#                 negative_neutral_reviews, neg_embs, neg_labels, neg_cents, request.top_representatives
#             )

#             for cluster_num, rep_list in neg_reps.items():
#                 for rep in rep_list[:2]:  # Take top 2 from each cluster
#                     reps_text.append(f"Cluster {cluster_num}: {rep}")
#             print(f"Selected representatives: {reps_text}")

#         else:
#             reps_text = cleaned[:10]  # fallback
#         # do negative clustering here
#         prompt = (
#             "Given the following customer reviews, generate one point for each cluster summarizing their main issue,:\n\n"
#             + "\n".join(f"- {r}" for r in reps_text[:10])
#             + "\n\nProblem:"
#         )
#         print(f"Generated prompt: {prompt}")

#         response = client.models.generate_content(
#         model="gemini-2.0-flash", contents=prompt
#         )


#         problem_statement = response.text
#         print(f"Generated problem statement: {problem_statement}")
#         return {"problem_statement": problem_statement}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Problem generation failed: {str(e)}")

import re

@app.post("/generate-problem")
async def generate_problem(request: ReviewRequest):
    try:
        if not request.raw_reviews:
            raise HTTPException(status_code=400, detail="No reviews provided")

        # Clean and score
        cleaned = [clean_text(r) for r in request.raw_reviews]
        sentiment_results = get_sentiment_scores(cleaned)

        # Separate by sentiment
        positive_reviews, negative_neutral_reviews = separate_by_sentiment(sentiment_results)

        reps_text = []
        if negative_neutral_reviews:
            neg_embs = get_embeddings(negative_neutral_reviews)
            neg_labels, neg_cents = cluster_embeddings(neg_embs, request.max_clusters)
            
            neg_reps = top_representatives(
                negative_neutral_reviews, neg_embs, neg_labels, neg_cents, request.top_representatives
            )
            counters = []

            for cluster_num, rep_list in neg_reps.items():
                counters.append(len(rep_list))
                for rep in rep_list[:2]:  # Top 2 from each cluster
                    reps_text.append(f"Cluster {cluster_num}: {rep}")
        else:
            reps_text = cleaned[:10]

        # Construct prompt
        prompt = (
            "Given the following customer reviews, generate one point for each cluster summarizing their main issue:\n\n"
            + "\n".join(f"- {r}" for r in reps_text[:10])
            + "\n\nProblem:"
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        raw_output = response.text

        # Extract structured data using regex
        clusters = []
     
        print("No structured clusters found, falling back to bullet points.")
        # fallback: treat lines starting with * as bullet points
        for line in raw_output.strip().splitlines():
            if line.strip().startswith("*"):
                clusters.append({
                    "cluster": len(clusters),
                    "summary": line.strip().lstrip("* ").strip()[11:]
                })
        print(f"Extracted clusters: {clusters}")
        return { "problem_statements": clusters,
                "counters": counters }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Problem generation failed: {str(e)}")


# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")