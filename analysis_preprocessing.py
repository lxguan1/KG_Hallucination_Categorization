import os
import json
from glob import glob
from collections import defaultdict

def load_all_classifications(folder_path):
    """
    Load and concatenate all `classifications` lists from JSON files in a folder.

    Args:
        folder_path (str): Path to the folder containing JSON files.

    Returns:
        list: Combined list of all classification entries.
    """
    all_classifications = []

    # Get all JSON file paths in the folder
    json_files = glob(os.path.join(folder_path, "*.json"))

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if "classifications" in data:
                    all_classifications.extend(data["classifications"])
                else:
                    print(f"'classifications' key not found in: {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return all_classifications

folder_path = "classified_new" 
all_classifications = load_all_classifications(folder_path)
print(f"Total classifications loaded: {len(all_classifications)}")

def split_by_hallucination_category(data_list, output_dir):
    """
    Splits a list of classification entries by hallucination category and saves them as separate files.

    Args:
        data_list (list): List of dictionaries, each with a 'classification' block.
        output_dir (str): Directory to save the split JSON files.
    """
    category_groups = defaultdict(list)

    # Group by hallucination_category
    for entry in data_list:
        category = entry.get("classification", {}).get("hallucination_category", "Unknown")
        category_groups[category].append(entry)

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save each group
    for category, entries in category_groups.items():
        safe_name = category.lower().replace(" ", "_").replace("/", "_")
        output_path = os.path.join(output_dir, f"{safe_name}.json")
        with open(output_path, "w") as f:
            json.dump(entries, f, indent=4)
        print(f"Saved {len(entries)} entries to {output_path}")

output_dir = "Analysis"
split_by_hallucination_category(all_classifications, output_dir)