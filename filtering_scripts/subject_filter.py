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
def filter_subject_group(subject, triples, model, z_thresh=0.0):
    sentences = [triple_to_sentence(subject, rel, obj) for (_, rel, obj) in triples]
    embeddings = model.encode(sentences)
    sim_matrix = util.cos_sim(embeddings, embeddings).numpy()
    np.fill_diagonal(sim_matrix, 0)
    mean_sims = sim_matrix.mean(axis=1)
    z_scores = (mean_sims - np.mean(mean_sims)) / np.std(mean_sims)
    
    kept = [triples[i] for i in range(len(triples)) if z_scores[i] > z_thresh]
    dropped = [triples[i] for i in range(len(triples)) if z_scores[i] <= z_thresh]
    return kept, dropped

# Save output
def save_filtered_triples(filepath, kept_triples):
    out_path = filepath.replace("graph_edgelist", "filtered_edgelist")
    with open(out_path, 'w') as f:
        for (e1, rel, e2) in kept_triples:
            line = f"{e1}|{e2}|{{'relation': '{rel}'}}\n"
            f.write(line)
    print(f"✅ Filtered file saved to: {out_path}")

# Run on a single file
if __name__ == "__main__":
    file_path = "edgelists/graph_edgelist_380"  # Update to test other files
    print(f"\n🔍 Loading file: {file_path}")
    triples = load_kg_file(file_path)
    print(f"Total triples loaded: {len(triples)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Group by subject
    grouped = defaultdict(list)
    for e1, rel, e2 in triples:
        grouped[e1].append((e1, rel, e2))

    all_kept = []
    all_dropped = []

    for subject, group in grouped.items():
        if len(group) < 3:
            # not enough to compare
            all_kept.extend(group)
            continue
        kept, dropped = filter_subject_group(subject, group, model)
        all_kept.extend(kept)
        all_dropped.extend(dropped)

    print(f"\n✅ Final kept triples: {len(all_kept)}")
    print(f"🗑️ Dropped as inconsistent: {len(all_dropped)}")

    save_filtered_triples(file_path, all_kept)