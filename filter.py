import ast
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load the file
def load_kg_file(filepath):
    triples = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split('|')
                if len(parts) != 3:
                    continue
                entity1, entity2, relation_str = parts
                relation = ast.literal_eval(relation_str).get('relation', '')
                triples.append((entity1.strip(), relation.strip(), entity2.strip()))
            except Exception as e:
                print(f"Skipping line due to error: {e}")
    return triples

# Convert triple to sentence
def triple_to_sentence(triple):
    e1, rel, e2 = triple
    return f"{e1} {rel} {e2}"

# Compute similarity and filter
def filter_triples(triples, z_thresh=-1.0):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = [triple_to_sentence(t) for t in triples]
    embeddings = model.encode(sentences)
    
    # Compute mean similarity per sentence
    similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()
    np.fill_diagonal(similarity_matrix, 0)  # ignore self-similarity
    mean_sims = similarity_matrix.mean(axis=1)

    # Compute z-scores
    z_scores = (mean_sims - np.mean(mean_sims)) / np.std(mean_sims)

    # Keep only non-outliers
    kept = [triples[i] for i in range(len(triples)) if z_scores[i] > z_thresh]
    dropped = [triples[i] for i in range(len(triples)) if z_scores[i] <= z_thresh]

    return kept, dropped

def save_filtered_triples(filepath, kept_triples):
    out_path = filepath.replace("graph_edgelist", "graph_edgelist_filtered")
    with open(out_path, 'w') as f:
        for (e1, rel, e2) in kept_triples:
            line = f"{e1}|{e2}|{{'relation': '{rel}'}}\n"
            f.write(line)
    print(f"\n💾 Filtered triples saved to: {out_path}")

if __name__ == "__main__":
    file_path = "edgelists/graph_edgelist_0"  # to change
    print(f"\nLoading file: {file_path}")
    triples = load_kg_file(file_path)
    print(f"Total triples: {len(triples)}")

    kept, dropped = filter_triples(triples)

    save_filtered_triples(file_path, kept)