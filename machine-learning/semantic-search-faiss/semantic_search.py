"""
Semantic Search Engine with FAISS + Sentence Transformers
=========================================================
Builds a fully local semantic search engine.
Requirements: pip install sentence-transformers faiss-cpu numpy rich matplotlib scikit-learn
"""
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

console = Console()

# 140 documents across 7 categories: Tech, Science, Cooking, Travel, Health, Business, Arts
documents = [
    "Python is a high-level programming language known for its readability and simplicity",
    "Docker containers package applications with their dependencies for consistent deployment",
    "REST APIs use HTTP methods like GET, POST, PUT, and DELETE to interact with web resources",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy",
    "Black holes are regions of spacetime where gravity is so strong that nothing can escape",
    "Plate tectonics explains how Earth's crust moves, causing earthquakes and volcanic activity",
    "Pasta carbonara is an Italian dish made with eggs, cheese, pancetta, and black pepper",
    "Sourdough bread uses naturally occurring wild yeast and bacteria for fermentation",
    "The Maillard reaction creates brown crusts and complex flavors when proteins are heated",
    "The Great Wall of China stretches over 13,000 miles across northern China",
    "Tokyo is the most populous metropolitan area in the world with over 37 million residents",
    "Bali is an Indonesian island known for its terraced rice paddies and Hindu temples",
    "Regular cardiovascular exercise strengthens the heart and improves blood circulation",
    "A balanced diet includes fruits, vegetables, whole grains, lean proteins, and healthy fats",
    "Meditation reduces stress by helping practitioners focus on the present moment",
    "Compound interest allows investments to grow exponentially over long periods of time",
    "Diversification spreads investment risk across different asset classes and sectors",
    "A budget helps individuals and businesses track income and expenses to meet financial goals",
    "The Renaissance was a period of great artistic and intellectual achievement in Europe",
    "Digital art uses computer technology as an essential part of the creative process",
    "Abstract art uses shapes, colors, and forms to achieve its effect rather than realistic depiction",
    "Git is a distributed version control system that tracks changes in source code",
    "Kubernetes orchestrates containerized applications across clusters of machines",
    "Neural networks are computing systems inspired by biological neurons in the human brain",
    "DNA molecules contain the genetic instructions for the development of all living organisms",
    "Evolution by natural selection explains how species adapt to their environments over time",
    "Climate change refers to long-term shifts in global temperatures and weather patterns",
    "Sushi is a Japanese dish of vinegared rice combined with raw fish and vegetables",
    "Chocolate chip cookies should be baked until the edges are golden but the center is soft",
    "Baking requires precise measurements because it involves complex chemical reactions",
    "Machu Picchu is a 15th-century Inca citadel located high in the Andes Mountains in Peru",
    "The Northern Lights are caused by solar particles interacting with Earth's magnetic field",
    "Iceland has over 130 volcanoes and numerous geothermal hot springs used for bathing",
    "Yoga combines physical postures, breathing techniques, and meditation for overall wellness",
    "Getting seven to nine hours of quality sleep each night is essential for cognitive function",
    "Strength training builds muscle mass and increases bone density, reducing injury risk",
    "The stock market enables companies to raise capital by selling shares to public investors",
    "Cryptocurrencies use cryptographic techniques to enable secure decentralized transactions",
    "Venture capital firms invest in early-stage companies with high growth potential",
    "Impressionist painters like Monet used loose brushstrokes to capture the effects of light",
    "Jazz music originated in African American communities in New Orleans in the early 1900s",
    "Hip hop culture emerged in the Bronx during the 1970s and includes rap, DJing, and breakdancing",
]

# Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype(np.float32))

def semantic_search(query: str, top_k: int = 5):
    """Search for documents semantically similar to the query."""
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query_embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({"score": float(score), "similarity_pct": f"{score * 100:.1f}%", "document": documents[idx]})
    return results

# Demo
console.print(Panel("[bold cyan]Semantic Search Demo[/bold cyan]", border_style="blue"))
queries = [
    "How do I make pasta at home?",
    "What causes earthquakes and volcanic eruptions?",
    "Tell me about investing and saving money",
    "Best places to visit in Asia",
    "How to stay healthy and fit",
    "I want to learn web development",
    "What is the theory of evolution?",
]

for query in queries:
    results = semantic_search(query, top_k=3)
    console.print(f"\n[bold]Query:[/bold] [cyan]{query}[/cyan]")
    for i, r in enumerate(results, 1):
        console.print(f"  {i}. ({r['similarity_pct']}) {r['document'][:80]}")

# Visualize with PCA
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings)
categories = ["Tech", "Science", "Cooking", "Travel", "Health", "Business", "Arts"]
colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#06b6d4", "#ec4899"]
fig, ax = plt.subplots(figsize=(14, 10))
docs_per_cat = len(documents) // len(categories)
for i, cat in enumerate(categories):
    mask = [j // docs_per_cat == i for j in range(len(documents))]
    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], c=colors[i], label=cat, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
ax.set_title("Document Embeddings Visualized with PCA\n384-dimensional vectors -> 2D projection", fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('embedding_visualization.png', dpi=150)
console.print("[green]Visualization saved![/green]")
