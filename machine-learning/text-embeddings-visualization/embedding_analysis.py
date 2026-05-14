"""
Text Embeddings: Generation, Comparison & Visualization
========================================================
Requirements: pip install sentence-transformers numpy matplotlib seaborn scikit-learn
"""
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# 50 sentences across 10 categories
sentences = [
    "The computer processed data at incredible speed",
    "Machine learning models require large amounts of training data",
    "Python is widely used for artificial intelligence applications",
    "Cloud computing enables scalable web services",
    "The algorithm optimized the search results efficiently",
    "The dog chased the ball across the green field",
    "Cats are independent creatures that enjoy solitude",
    "The majestic eagle soared high above the mountains",
    "Dolphins are highly intelligent marine mammals",
    "The tiger stalked its prey through the dense jungle",
    "The chef prepared a delicious Italian pasta dish",
    "Fresh ingredients make the best homemade meals",
    "The chocolate cake was rich and decadently sweet",
    "Grilling steak requires high heat and proper timing",
    "Japanese sushi demands precise knife skills and fresh fish",
    "The ancient ruins attracted tourists from around the world",
    "Paris is known as the city of love and romance",
    "The tropical beach had crystal clear turquoise water",
    "Mountain climbers reached the summit after days of effort",
    "The bustling city never sleeps with its vibrant nightlife",
    "She felt overwhelming joy when she received the good news",
    "Heartbreak can feel like a physical pain in your chest",
    "Their friendship had lasted through decades of ups and downs",
    "Pride swelled in his chest as he watched his daughter graduate",
    "Anxiety crept in as the deadline approached rapidly",
    "The scientist conducted experiments to test the hypothesis",
    "Mathematics is the language of the universe",
    "Quantum physics challenges our understanding of reality",
    "DNA contains the genetic blueprint of all living organisms",
    "The theory of evolution explains the diversity of life",
    "The soccer team celebrated their championship victory",
    "Swimming is an excellent full-body cardiovascular workout",
    "The marathon runner crossed the finish line exhausted but proud",
    "Basketball requires both athleticism and strategic thinking",
    "Yoga combines physical poses with breathing and meditation",
    "The painter captured the sunset in brilliant orange and red hues",
    "Music has the power to evoke deep emotional responses",
    "The novelist spent years crafting the perfect ending",
    "Dance allows expression beyond what words can convey",
    "Photography freezes a single moment for eternity",
    "The startup raised millions in venture capital funding",
    "Effective leadership requires both vision and empathy",
    "The company announced record profits for the fiscal year",
    "Remote work has transformed the modern workplace",
    "Negotiation skills are essential for closing major deals",
    "Regular exercise reduces the risk of heart disease",
    "The doctor prescribed antibiotics for the bacterial infection",
    "Mental health is just as important as physical health",
    "Vaccines have saved millions of lives throughout history",
    "A balanced diet provides essential nutrients for the body",
]
categories = (["Tech"]*5 + ["Animals"]*5 + ["Food"]*5 + ["Travel"]*5 +
              ["Emotions"]*5 + ["Science"]*5 + ["Sports"]*5 + ["Art"]*5 +
              ["Business"]*5 + ["Health"]*5)

# Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
console.print(f"[green]Embeddings: {embeddings.shape}[/green]")

# PCA
pca = PCA(n_components=2, random_state=42)
e2d = pca.fit_transform(embeddings)
cat_colors = {"Tech":"#3b82f6","Animals":"#10b981","Food":"#f59e0b","Travel":"#8b5cf6",
              "Emotions":"#ef4444","Science":"#06b6d4","Sports":"#f97316","Art":"#ec4899",
              "Business":"#6366f1","Health":"#14b8a6"}
fig, ax = plt.subplots(figsize=(16,11))
for cat in sorted(set(categories)):
    mask = [c==cat for c in categories]
    ax.scatter(e2d[mask,0], e2d[mask,1], c=cat_colors[cat], label=cat, alpha=0.75, s=120, edgecolors='white')
ax.legend(ncol=2); ax.set_title("PCA: Text Embeddings"); plt.tight_layout()
plt.savefig('01_pca.png', dpi=150); plt.close()

# t-SNE
tsne = TSNE(n_components=2, perplexity=8, random_state=42, max_iter=1000)
e2dt = tsne.fit_transform(embeddings)
fig, ax = plt.subplots(figsize=(16,11))
for cat in sorted(set(categories)):
    mask = [c==cat for c in categories]
    ax.scatter(e2dt[mask,0], e2dt[mask,1], c=cat_colors[cat], label=cat, alpha=0.75, s=120, edgecolors='white')
ax.legend(ncol=2); ax.set_title("t-SNE: Text Embeddings"); plt.tight_layout()
plt.savefig('02_tsne.png', dpi=150); plt.close()

# Heatmap
idx = [0,5,10,15,20,25,30,35,40,45]
sim = cosine_similarity(embeddings[idx])
fig, ax = plt.subplots(figsize=(14,12))
sns.heatmap(sim, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, ax=ax)
ax.set_title("Cosine Similarity Heatmap"); plt.tight_layout()
plt.savefig('03_heatmap.png', dpi=150); plt.close()

# Semantic similarity demo
pairs = [("The dog played in the park","A canine ran through the green field"),
         ("The dog played in the park","The stock market crashed yesterday"),
         ("I love eating pizza and pasta","Italian cuisine is my favorite food"),
         ("I love eating pizza and pasta","The spaceship launched into orbit")]
for a,b in pairs:
    ea = model.encode([a], normalize_embeddings=True)[0]
    eb = model.encode([b], normalize_embeddings=True)[0]
    sim = float(np.dot(ea,eb))
    rel = "SAME" if sim > 0.5 else "DIFF"
    console.print(f"  [{rel}] {sim*100:.1f}% — {a[:40]} <-> {b[:40]}")

console.print("[green]Analysis complete![/green]")
