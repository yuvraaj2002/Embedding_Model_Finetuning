# Embedding Model Fine-Tuning Project: Complete Action Plan

## Phase 0: Prerequisites & Foundational Knowledge (Week 1)

### 0.1 Understand Embeddings Fundamentals
**What to learn:**
- What embeddings are: Dense vector representations of text (typically 384-1536 dimensions)
- How they work: Text → Tokenization → Neural Network → Vector output
- Distance metrics: Cosine similarity, Euclidean distance, dot product
- Why fine-tune: Pre-trained embeddings may not capture domain-specific relationships

**Resources:**
- Read: https://huggingface.co/blog/getting-started-with-embeddings
- Read: "Sentence-Transformers: Sentence Embeddings using Siamese BERT-Networks" paper
- Watch: YouTube - "What are Embeddings?" (StatQuest or similar)

**Time investment:** 4-6 hours
**Deliverable:** Understand why embeddings work and when you need to fine-tune

---

### 0.2 Choose Your Base Model
**Three popular options:**

| Model | Dimensions | Speed | Accuracy | Best For |
|-------|-----------|-------|----------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | Quick prototyping, limited compute |
| all-mpnet-base-v2 | 768 | Medium | Better | General-purpose, balanced |
| all-roberta-large-v1 | 1024 | Slower | Best | When accuracy is critical |

**Decision framework:**
- **Limited GPU resources?** → Use all-MiniLM-L6-v2
- **Want good balance?** → Use all-mpnet-base-v2
- **Have GPU resources & accuracy matters?** → Use all-roberta-large-v1

**Implementation:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')  # Or your chosen model
```

---

### 0.3 Set Up Development Environment
**Install these tools:**
```bash
# Create virtual environment
python -m venv embedding_env
source embedding_env/bin/activate  # On Windows: embedding_env\Scripts\activate

# Install core packages
pip install sentence-transformers torch transformers datasets scikit-learn pandas numpy

# For evaluation
pip install scipy matplotlib seaborn

# For GPU support (CUDA 11.8+)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Verify setup:**
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## Phase 1: Data Preparation (Week 2)

### 1.1 Define Your Use Case
**Pick one specific problem to solve:**

Examples:
- E-commerce: Find similar products by description
- Search: Match user queries to relevant documents
- Clustering: Group customer support tickets by issue type
- Recommendation: Find related articles/posts

**Be specific:** "I want embeddings that can match customer support queries to KB articles with 90% accuracy"

---

### 1.2 Gather Training Data
**What you need:**
- Pairs of similar sentences/documents (positive pairs)
- Ideally triplets: (anchor, positive, negative) for better training
- Minimum: 500 pairs; Better: 5,000+; Ideal: 10,000+

**Data collection strategies:**

**Option A: Use existing datasets (easiest)**
```python
# Download from Hugging Face
from datasets import load_dataset

# Example: STS Benchmark for semantic similarity
dataset = load_dataset("stsb_multi_mt", "en")
```

**Option B: Create from your domain (better results)**
- Find Q&A pairs in your documentation
- Extract similar documents from your database
- Use weak labels: "user searched for X and clicked document Y" = similar pair
- Manual annotation: Tag 1000+ pairs yourself or crowdsource

**Option C: Synthetic data (scalable)**
```python
# Use Claude API to generate similar sentence pairs
# Prompt: "Generate 5 variations of this technical support query: '{query}'"
# This amplifies your limited data
```

**Format your data as CSV:**
```csv
sentence1,sentence2,label
"How do I reset my password?","What's the process to change my account password?",1
"How do I reset my password?","What payment methods do you accept?",0
```

**or as JSONL (better for complex data):**
```jsonl
{"texts": ["Query about password", "FAQ on password reset"], "label": 1}
{"texts": ["Query about password", "FAQ on billing"], "label": 0}
```

---

### 1.3 Split & Validate Data
**Split your data:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('training_data.csv')

# 70/10/20 split: train/val/test
train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.67, random_state=42)

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
```

**Validate quality:**
- Check no data leakage (same pairs in train/test)
- Verify label distribution (roughly balanced)
- Sample 20 pairs manually - do they make sense?

---

### 1.4 Create Data Loaders
```python
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

def create_examples(df):
    examples = []
    for idx, row in df.iterrows():
        examples.append(InputExample(
            texts=[row['sentence1'], row['sentence2']],
            label=float(row['label'])
        ))
    return examples

train_examples = create_examples(train)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
```

---

## Phase 2: Fine-Tuning (Week 3)

### 2.1 Choose Loss Function
**Different objectives, different losses:**

| Loss Function | Use Case | Code |
|---|---|---|
| **CosineSimilarityLoss** | Ranking, similarity scores | `losses.CosineSimilarityLoss(model)` |
| **ContrastiveLoss** | Binary similarity (similar/dissimilar) | `losses.ContrastiveLoss(model)` |
| **TripletLoss** | Triplets (anchor, positive, negative) | `losses.TripletLoss(model)` |
| **MultipleNegativesRankingLoss** | Hard negative mining (best) | `losses.MultipleNegativesRankingLoss(model)` |

**Recommendation:**
- Start with **CosineSimilarityLoss** if you have 0-1 similarity scores
- Use **MultipleNegativesRankingLoss** if you have minimal data (it's more efficient)

---

### 2.2 Set Up Training
```python
from sentence_transformers import models, losses, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
import math

# Initialize model
model = SentenceTransformer('all-mpnet-base-v2')

# Choose loss function
train_loss = losses.CosineSimilarityLoss(model)

# Create evaluator for validation set
sentences1 = val['sentence1'].tolist()
sentences2 = val['sentence2'].tolist()
scores = val['label'].tolist()

evaluator = EmbeddingSimilarityEvaluator(
    sentences1, sentences2, scores,
    main_similarity='cosine'
)

# Set up training arguments
training_args = SentenceTransformerTrainingArguments(
    output_dir="./embedding_model_output",
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    logging_steps=10,
    seed=42,
)

# Create trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_examples,
    loss=train_loss,
    evaluator=evaluator,
    compute_metrics_fn=None,
)
```

---

### 2.3 Train the Model
```python
# Start training
trainer.train()

# Save the fine-tuned model
model.save('my-finetuned-embedding-model')
```

**What to expect:**
- Training time: 1-4 hours on GPU (depends on dataset size)
- Loss should decrease smoothly
- Validation similarity should improve

**Monitor these metrics:**
- **Training loss**: Should decrease steadily
- **Validation loss**: Should decrease (if it increases, you're overfitting)
- **Learning rate**: Typically 2e-5 works well; if loss is erratic, try 1e-5

**Common issues & fixes:**
- Loss not decreasing? → Lower learning rate or check data quality
- GPU out of memory? → Reduce batch_size (16 → 8)
- Overfitting? → Add weight_decay, reduce num_train_epochs, or get more data

---

## Phase 3: Evaluation (Week 3-4)

### 3.1 Evaluate on Test Set
```python
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

# Load your fine-tuned model
finetuned_model = SentenceTransformer('./my-finetuned-embedding-model')
original_model = SentenceTransformer('all-mpnet-base-v2')

# Get embeddings
test_embeddings_finetuned = finetuned_model.encode(test['sentence1'].tolist())
test_embeddings_original = original_model.encode(test['sentence1'].tolist())

# Compare with test labels
from sklearn.metrics.pairwise import cosine_similarity

def evaluate(embeddings, test_df):
    similarities = []
    for i, row in test_df.iterrows():
        # Get embeddings for both sentences
        emb1 = embeddings[i]
        emb2 = finetuned_model.encode(row['sentence2'])
        
        # Calculate similarity
        sim = cosine_similarity([emb1], [emb2])[0][0]
        similarities.append(sim)
    
    # Calculate Spearman correlation with ground truth
    correlation, p_value = spearmanr(test_df['label'], similarities)
    return correlation

# Compare
original_score = evaluate(test_embeddings_original, test)
finetuned_score = evaluate(test_embeddings_finetuned, test)

print(f"Original model Spearman correlation: {original_score:.4f}")
print(f"Fine-tuned model Spearman correlation: {finetuned_score:.4f}")
print(f"Improvement: {((finetuned_score - original_score) / abs(original_score) * 100):.2f}%")
```

### 3.2 Create Domain-Specific Benchmark
```python
# If you have specific examples that matter most:
critical_pairs = [
    ("reset password", "how do i change my password", 1),
    ("billing question", "how to upgrade plan", 0),
    # ... add 20-50 critical domain pairs
]

def test_critical_pairs(model, pairs):
    correct = 0
    for text1, text2, expected_similar in pairs:
        emb1 = model.encode(text1)
        emb2 = model.encode(text2)
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        
        # Consider similar if > 0.5
        predicted_similar = similarity > 0.5
        if predicted_similar == expected_similar:
            correct += 1
    
    accuracy = correct / len(pairs)
    print(f"Critical pairs accuracy: {accuracy:.2%}")
    return accuracy

test_critical_pairs(finetuned_model, critical_pairs)
```

### 3.3 Test Real-World Scenarios
```python
# Scenario 1: Similarity search
query = "How do I reset my password?"
similar_docs = [
    "Password reset instructions",
    "Account security settings",
    "Billing FAQ",
    "Create new account",
]

query_embedding = finetuned_model.encode(query)
doc_embeddings = finetuned_model.encode(similar_docs)

similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
ranked = sorted(zip(similar_docs, similarities), key=lambda x: x[1], reverse=True)

print("Top matches:")
for doc, score in ranked[:3]:
    print(f"  {doc}: {score:.3f}")
```

---

## Phase 4: Deployment & Optimization (Week 4)

### 4.1 Quantization for Speed
```python
# Make model faster without much accuracy loss
from sentence_transformers import util

# Quantize to int8
model_quantized = util.quantize_embeddings(
    finetuned_model.encode(test['sentence1'].tolist()[:100])
)

# Result: 4x smaller embeddings, slightly faster computation
```

### 4.2 Create Production Inference Script
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingService:
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)
    
    def embed_text(self, text):
        """Convert text to embedding"""
        return self.model.encode(text)
    
    def find_similar(self, query, documents, top_k=5):
        """Find top-k similar documents to query"""
        query_embedding = self.model.encode(query)
        doc_embeddings = self.model.encode(documents)
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            {"document": documents[i], "similarity": similarities[i]}
            for i in top_indices
        ]

# Usage
service = EmbeddingService('./my-finetuned-embedding-model')
results = service.find_similar(
    "How do I reset my password?",
    ["Password reset guide", "Account settings", "Billing info"],
    top_k=2
)
print(results)
```

### 4.3 Save Model for Production
```python
# Save for Hugging Face Hub (optional, to share)
model.push_to_hub("your-username/my-embedding-model")

# Or save locally
model.save('./production-embedding-model')

# Load in production
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('./production-embedding-model')
```

---

## Phase 5: Advanced Optimization (Optional, Week 5+)

### 5.1 If Accuracy Still Needs Improvement
**Option A: Hard Negative Mining**
```python
# Use MultipleNegativesRankingLoss - forces model to distinguish hard cases
train_loss = losses.MultipleNegativesRankingLoss(model)
```

**Option B: Get More Data**
- Even 2-3x more training data can significantly help
- Target: 10,000+ examples for robust model

**Option C: Longer Training**
```python
# Increase epochs
training_args.num_train_epochs = 8  # Instead of 4
```

**Option D: Knowledge Distillation**
```python
# Train a smaller model to match your fine-tuned model's outputs
# Makes your model faster for deployment
from sentence_transformers import losses

distillation_loss = losses.MatryoshkaLoss(model, loss_fct=train_loss)
```

### 5.2 Domain-Specific Fine-Tuning
If you have domain experts who can create labeled data:
- Get 1,000 high-quality labeled pairs from your domain
- Fine-tune again on top of your current model
- Usually gives another 5-15% improvement

---

## Phase 6: Integration & Monitoring

### 6.1 Integration Points
- **Search engines**: Index documents with embeddings, use for semantic search
- **Recommendation systems**: Find similar products/content
- **Clustering**: Group customer tickets automatically
- **Classification**: Use embeddings + simple classifier for domain classification

### 6.2 Monitor in Production
```python
import json
from datetime import datetime

class EmbeddingMonitor:
    def __init__(self):
        self.metrics = []
    
    def log_prediction(self, query, top_matches, confidence):
        self.metrics.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "top_matches": top_matches,
            "confidence": confidence,
        })
    
    def detect_drift(self):
        """Check if model performance degrades over time"""
        # Compare recent embeddings to earlier embeddings
        # If similarity distributions change significantly, retrain
        pass
    
    def export_metrics(self):
        with open('embedding_metrics.jsonl', 'w') as f:
            for metric in self.metrics:
                f.write(json.dumps(metric) + '\n')
```

---

## Success Metrics & Milestones

| Milestone | Target | How to Measure |
|-----------|--------|---|
| **Week 1** | Understand embeddings, set up environment | Can explain embeddings, run test inference |
| **Week 2** | Gather 1,000+ training pairs | Have clean training/val/test splits |
| **Week 3** | Complete fine-tuning, run evaluation | Fine-tuned model shows 10%+ improvement |
| **Week 4** | Deploy production version | Inference script works, tested on 50+ examples |
| **Week 5+** | Iterate & optimize | Model accuracy meets 90%+ threshold on critical use cases |

---

## Tech Stack Summary

**Programming:** Python 3.8+
**Core Libraries:**
- `sentence-transformers`: Fine-tuning & inference
- `torch`: Deep learning backend
- `transformers`: Pre-trained models
- `scikit-learn`: Evaluation metrics
- `pandas`: Data handling

**Hardware recommendations:**
- GPU: NVIDIA GPU with 6GB+ VRAM (RTX 3060, A100, etc.)
- CPU: Modern CPU (Intel i7/i9 or AMD Ryzen 7+)
- RAM: 16GB+
- Storage: 10GB for model + data

**Alternative (if no GPU):** Use `sentence-transformers` on CPU - slower but works

---

## Common Pitfalls to Avoid

1. **Garbage in, garbage out**: Bad training data = bad model. Spend 40% of time on data quality.
2. **Overfitting on small datasets**: If you have <500 pairs, use pre-trained model as-is or try data augmentation.
3. **Not evaluating properly**: Test on data your model has never seen; check for data leakage.
4. **Ignoring class imbalance**: If 90% of pairs are "similar", model learns to always predict similar.
5. **Training too long**: Monitor validation loss; stop when it plateaus or increases.
6. **Choosing wrong loss function**: CosineSimilarityLoss for scores, MultipleNegativesRankingLoss for ranking.
7. **Not comparing to baseline**: Always measure improvement vs. original pre-trained model.

---

## Resources & Links

**Learning:**
- Hugging Face Sentence-Transformers docs: https://www.sbert.net/
- Understanding embeddings: https://huggingface.co/blog/getting-started-with-embeddings
- Papers on fine-tuning: "Sentence-BERT" (Reimers & Gupta, 2019)

**Code Examples:**
- Official Sentence-Transformers repo: https://github.com/UKPLab/sentence-transformers
- Fine-tuning examples: https://www.sbert.net/docs/training/overview.html

**Data:**
- Hugging Face Datasets: https://huggingface.co/datasets
- STS Benchmark: https://www.sbert.net/docs/datasets/stsbenchmark.html

---

## Next Steps (Right Now)

1. **Tomorrow:** Read the Sentence-Transformers documentation (2 hours)
2. **This week:** Set up your environment and run a test fine-tuning on public data
3. **Next week:** Collect or prepare YOUR domain-specific data
4. **Following week:** Begin fine-tuning on real data

Good luck! This is a completely achievable project. Start with Phase 0, move sequentially, and don't skip the evaluation phase.