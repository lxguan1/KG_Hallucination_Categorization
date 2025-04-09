import os
import ast
import numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

# Load triples from file
def load_kg_file(filepath):
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

# Filter a group of triples for a fixed subject
def filter_subject_group(subject, triples, model, z_thresh=2.0):
    sentences = [triple_to_sentence(subject, rel, obj) for (_, rel, obj) in triples]
    embeddings = model.encode(sentences)
    sim_matrix = util.cos_sim(embeddings, embeddings).numpy()
    np.fill_diagonal(sim_matrix, 0)
    mean_sims = sim_matrix.mean(axis=1)
    z_scores = (mean_sims - np.mean(mean_sims)) / np.std(mean_sims)
    
    kept = [triples[i] for i in range(len(triples)) if z_scores[i] > z_thresh]
    dropped = [triples[i] for i in range(len(triples)) if z_scores[i] <= z_thresh]
    return kept, dropped

# Save output to filtered_edgelists/
def save_filtered_triples(filename, kept_triples):
    out_path = os.path.join("filtered_edgelists", filename)
    with open(out_path, 'w') as f:
        for (e1, rel, e2) in kept_triples:
            line = f"{e1}|{e2}|{{'relation': '{rel}'}}\n"
            f.write(line)
    print(f"✅ Saved to: {out_path}")

# Main
if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    edgelist_dir = "edgelists"
    output_dir = "filtered_edgelists"

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(edgelist_dir):
        if not filename.startswith("graph_edgelist_"):
            continue

        file_path = os.path.join(edgelist_dir, filename)
        print(f"\n Processing: {file_path}")
        triples = load_kg_file(file_path)
        print(f" Total triples loaded: {len(triples)}")

        # Group by subject
        grouped = defaultdict(list)
        for e1, rel, e2 in triples:
            grouped[e1].append((e1, rel, e2))

        all_kept = []
        all_dropped = []

        for subject, group in grouped.items():
            if len(group) < 3:
                all_kept.extend(group)
                continue
            kept, dropped = filter_subject_group(subject, group, model, z_thresh=2.0)
            all_kept.extend(kept)
            all_dropped.extend(dropped)

        print(f" Final kept: {len(all_kept)} |  Dropped: {len(all_dropped)}")

        save_filtered_triples(filename.replace("graph_", "filtered_"), all_kept)