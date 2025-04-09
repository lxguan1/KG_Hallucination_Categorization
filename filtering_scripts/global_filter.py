import ast
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# Load triples
def load_triples(filepath):
    triples = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split('|')
                if len(parts) != 3:
                    continue
                e1, e2, rel_str = parts
                relation = ast.literal_eval(rel_str).get('relation', '').strip()
                triples.append((e1.strip(), relation, e2.strip()))
            except:
                continue
    return triples

# Convert to natural language
def triple_to_sentence(e1, rel, e2):
    return f"{e1} {rel} {e2}"

# Compute top-K average similarity per triple (faster and memory efficient)
def compute_avg_topk_similarities(embeddings, top_k=50):
    sim_scores = []
    for i in range(len(embeddings)):
        sims = util.cos_sim(embeddings[i], embeddings)[0]
        topk = torch.topk(sims, k=top_k+1)  # +1 because it includes itself
        topk_values = topk.values[1:]  # exclude self
        sim_scores.append(topk_values.mean().item())
    return np.array(sim_scores)

# Save final filtered triples
def save_triples(path, triples):
    with open(path, 'w') as f:
        for e1, rel, e2 in triples:
            f.write(f"{e1}|{e2}|{{'relation': '{rel}'}}\n")

# Run global filtering
if __name__ == "__main__":
    input_path = "combined_graph_edgelist.txt"
    output_path = "final_filtered.txt"
    z_thresh = 2.0
    top_k_neighbors = 50

    print(f" Loading triples from {input_path}...")
    triples = load_triples(input_path)
    print(f"Loaded {len(triples)} triples.")

    print(" Converting triples to sentences...")
    sentences = [triple_to_sentence(*t) for t in triples]

    print(" Encoding with SentenceTransformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, batch_size=128, show_progress_bar=True, convert_to_tensor=True)

    print(f" Calculating avg top-{top_k_neighbors} similarities...")
    sim_scores = compute_avg_topk_similarities(embeddings, top_k=top_k_neighbors)

    z_scores = (sim_scores - np.mean(sim_scores)) / np.std(sim_scores)
    kept = [triples[i] for i in range(len(triples)) if z_scores[i] > z_thresh]
    dropped = [triples[i] for i in range(len(triples)) if z_scores[i] <= z_thresh]

    print(f"\n Kept: {len(kept)}")
    print(f" Dropped: {len(dropped)}")

    save_triples(output_path, kept)
    print(f" Saved final output to: {output_path}")