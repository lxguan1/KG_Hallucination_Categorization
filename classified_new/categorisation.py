import json
from pathlib import Path
import time
from transformers import pipeline
import torch
import re
import argparse
from glob import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--start_index", type=int, required=True)
parser.add_argument("--end_index", type=int, required=True)
args = parser.parse_args()

def load_all_prompts_from_folder(folder_path: str):
    """Load and combine all prompt data from all .json files in a folder."""
    all_prompts = []

    for file_path in glob(os.path.join(folder_path, "*.json")):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                if "prompts" in data:
                    # print(f"Loading prompts from {file_path}")
                    # print(f"Number of prompts: {len(data['prompts'])}")
                    all_prompts.extend(data["prompts"])
                    # print(f"Total prompts loaded so far: {len(all_prompts)}")
                else:
                    print(f"Warning: No 'prompts' key in {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return all_prompts

folder_path = "outputs_new"
all_prompts = load_all_prompts_from_folder(folder_path)

print(f"Loaded {len(all_prompts)} prompts from folder: {folder_path}")
    
def frame_prompt(diamond_key, prompt, explanation):
    
    messages = [
        {
            "role": "user",
            "content": """
You are an expert in analyzing hallucinations in large language models (LLMs).

Your task is to read a model-generated response (including its input prompt and reasoning), determine whether the model hallucinated, and if so, classify the hallucination into one of the standard categories described below.

You must:
1. Carefully compare the model's answer and reasoning with the given prompt information.
2. check if llm is actually halucinating. if the explanation mentions that we cannot infer the facts, then the model is correct — there is no hallucination.
3. If there is hallucination, choose correct hallucination category based on the definitions and examples provided.
4. Provide a short explanation for your classification decision.

Here are the types of hallucinations and examples to guide your classification:

---

1. Intrinsic Hallucination
Definition: The model generates a claim that contradicts information explicitly provided in the input.
Example
Path 1
- Senator Adams opposed Bill 42.
- Bill 42 was designed to increase surveillance.
Path 2
- Senator Adams voted against increased government surveillance.
- Senator Adams criticized Homeland Security expansion.
Question
- Senator Adams supported Bill 42.
- Senator Adams helped draft Bill 42.
Explanation: We can infer from the facts that Adams likely supported the bill because he aligns with security policies.
-> This contradicts the prompt: Adams opposed the bill.

2. Extrinsic Hallucination
Definition: The model introduces new information not present or verifiable in the input.
Example
Path 1
- Louisiana experienced a hurricane.
- Emergency shelters opened in Baton Rouge.
Path 2
- National Guard deployed to evacuation points.
- Families were relocated to safer areas.
Question
- FEMA declared a national emergency.
- FEMA provided housing assistance.
Explanation: We can infer from the facts that FEMA likely intervened, as it usually does during disasters.
-> FEMA is never mentioned — this is unsupported but plausible.

3. Commonsense Hallucination
Definition: The model uses general world knowledge or expectations, not prompt content.
Example
Path 1
- Detroit automakers laid off workers.
- Unemployment rose 8%.
Path 2
- Foreclosures increased.
- Jobless claims hit a record.
Question
- Homelessness surged.
- Crime increased.
Explanation: We can infer from the facts that economic downturns often lead to homelessness and crime.
-> Reasonable, but not stated in the prompt.

4. Temporal Hallucination
Definition: The model infers an incorrect order of events.
Example
Path 1
- Evidence submitted in October.
- Committee convened in August.
Path 2
- FBI launched investigation in September.
- Chair requested more documents.
Question
- The FBI influenced the August meeting.
- The witness testimony was discussed in August.
Explanation: We can infer from the facts that the FBI investigation likely influenced the meeting.
-> Wrong order — events happened after the meeting.

5. Symmetric Completion
Definition: The model wrongly assumes a bidirectional relationship.
Example
Path 1
- Alice mentored Bob.
- Bob thanked Alice in public.
Path 2
- Bob wrote a book.
- Alice endorsed it.
Question
- Bob mentored Alice.
- Alice was Bob’s student.
Explanation: We can infer from the facts that mentorships are often mutual.
-> Only one direction is given; symmetry is assumed.

6. Attribute Transfer
Definition: The model assigns properties of one entity to another nearby one.
Example
Path 1
- Dr. Stevens received NIH funding.
- She works on mRNA vaccines.
Path 2
- Dr. Clark co-authored papers with Stevens.
- Clark works in biotech.
Question
- Clark is NIH-funded.
- Clark studies mRNA vaccines.
Explanation: We can infer from the facts that collaborators often have similar funding.
-> No support for Clark — traits are borrowed from Stevens.

7. Path Extrapolation
Definition: The model creates a shortcut between nodes using graph structure analogies.
Example
Path 1
- Johnson sponsored a bill.
- The bill came from the tech committee.
Path 2
- Lin chaired the committee.
- Lin debated Johnson.
Question
- Lin co-sponsored the bill.
- Lin voted for it.
Explanation: We can infer from the facts that they are clearly linked via the committee.
-> Shortcut is inferred, not given.

8. Misleading Prior (Training Bias)
Definition: The model responds with memorized or popular facts, overriding the prompt.
Example
Path 1
- ZetaTech launched a blockchain product.
- Based in Palo Alto.
Path 2
- Raised Series A from angel investors.
- First product shipped in 2024.
Question
- ZetaTech is a unicorn.
- It was backed by Sequoia.
Explanation: We can infer from the facts that this is typical for Palo Alto startups.
-> Prompt never says this — model is relying on learned priors.

9. Other / Unclassified
Definition: The hallucination does not clearly fit above categories or blends multiple issues.
Example
Path 1
- Witness was confused during testimony.
- Judge called a recess.
Path 2
- Jury asked for clarification.
- Prosecutor presented new evidence.
Question
- Witness was charged with perjury.
- Jury doubted their testimony.
Explanation: We can infer from the facts that inconsistencies often lead to legal consequences.
-> Ambiguous mix of reasoning — label as Other.

10. No Hallucination
Definition: The model’s response correctly states that the given facts cannot be inferred from the input.
Example
Path 1
- Senator Adams opposed Bill 42.
- Bill 42 was designed to increase surveillance.
Path 2
- Senator Adams voted against increased government surveillance.
Question
- Senator Adams supported Bill 42.
Explanation: We cannot infer from the facts that Senator Adams supported the bill.
-> No hallucination occurred.

"""
        },
        {
            "role": "user",
            "content": f"""Read the following LLM output and determine whether the model hallucinated. If the explanation says that we **cannot** infer the facts, then the model is correct — there is no hallucination. In that case, respond:

```json
{{
  "hallucination_category": "No Hallucination",
  "category_explanation": "The model correctly stated that the given facts cannot be inferred from the provided context. No hallucination occurred."
}}
        else, classify the following into appropriate hallucination basd on the definations given before:
        - Prompt: {prompt}
        - Explanation: {explanation}
        
        Classify the hallucination based on the prompt and category generated and reason, and give the category of hallucination, and also a reason why you are classifying that way.
        Your output should be in the form of a json object containing:
            ```json
            {{'hallucination_category': '<category>',
            'category_explanation': '<explanation of why that hallucinated category>'}}
        """
        }
    ]
    return messages

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

llama = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


# if "we cannot infer" in explanation.lower():
#     result = {
#         "hallucination": False,
#         "category": None,
#         "justification": "The model correctly stated that the given facts cannot be inferred from the provided context. No hallucination occurred."
#     }
#     print(json.dumps(result, indent=4))
# else:
#     messages = frame_prompt(diamond_key, prompt, explanation)
#     response = llama(messages, max_new_tokens=512)
#     output_text = response[0]['generated_text'][-1]['content']
#     print(f"Model output: {output_text}")

all_classifications = []
for sample_prompt in all_prompts[args.start_index:args.end_index]:
    print(f"Processing index {all_prompts.index(sample_prompt)}: {sample_prompt['diamond_key']}...")
    diamond_key = sample_prompt["diamond_key"]
    prompt = sample_prompt["prompt"]
    explanation = sample_prompt["explanation"]
    
    messages = frame_prompt(diamond_key, prompt, explanation)
    response = llama(messages, max_new_tokens=512)
    output_text = response[0]['generated_text'][-1]['content']
    
    match = re.search(r"```json\s*(\{.*?\})\s*```", output_text, re.DOTALL)
    
    if match:
        try:
            json_data = json.loads(match.group(1))  # Convert JSON string to dict
            all_classifications.append({
                "diamond_key": diamond_key,
                "prompt": prompt,
                "explanation": explanation,
                "classification": json_data
            })
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON for diamond {diamond_key}: {e}")
    else:
        print(f"No valid JSON block found for diamond {diamond_key}.")

# Write all classifications to a classifications.json file
with open(f"classifications_{args.start_index}_{args.end_index}.json", "w") as f:
    json.dump({"classifications": all_classifications}, f, indent=4)

