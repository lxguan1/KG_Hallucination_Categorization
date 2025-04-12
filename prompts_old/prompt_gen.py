import json
from pathlib import Path
import time
from transformers import pipeline
import torch
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start_index", type=int, required=True)
parser.add_argument("--end_index", type=int, required=True)
args = parser.parse_args()

file_path = Path("filtered_diamonds2.json")
with open(file_path, "r") as f:
    data = json.load(f)

diamonds_list = data.get("diamonds", [])
selected_diamonds = diamonds_list[args.start_index:args.end_index]

diamonds_dict = {
    f"diamond_{args.start_index + i}": d
    for i, d in enumerate(selected_diamonds)
}

def format_for_llama_prompt(diamond, partial_len=2):
    def to_text(edges):
        return "\n".join(f"{e['source']} {e['relation']} {e['target']}." for e in edges)

    branch1_text = to_text(diamond["branch1"]["edges"])
    branch2_text = to_text(diamond["branch2"]["edges"])
    terminal = diamond["terminal"]

    print("Branch 1 Text:")
    print(branch1_text)
    print("Branch 2 Text:")
    print(branch2_text)
    
    messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that writes questions as per the given context"
    },
    {
        "role": "user",
        "content": """You are given a diamond pattern in a knowledge graph.

- Branch 1 (Background):  
Greta Thunberg led a school strike.  
The school strike gained global media attention.  
This attention influenced environmental policy.  
Environmental policy relates to climate change.

- Branch 2 (Partial Information):  
Greta Thunberg gave a speech at the UN Youth Summit.  
The summit was attended by António Guterres.
António Guterres emphasized youth involvement in global leadership.
Youth involvement relates to climate change.

Now, write a True/False/Cannot_infer-style question based on this setup. The question should be made by giving the entire first path, and then first 2 relations of the second path as to be considered as complete knowledge, and ask a question by giving remaining relations of the second branch and asking if the model is able to tell if the remaining part given is true based on the given branch 1 and partial branch 2 or if its false, or if there is no sufficient knowledge given based on which we can infer. Always end your question in these two lines: \n\nAnswer using only one of the following options:\nTRUE\nFALSE\nCannot conclude from the given context.\n Selecting True/False means that there is enough information from path 1 and path 2, to conclude if the asked relations in the question are true or false.
Also make sure that the prompt has the senetence 'paqth 1' and 'path 2' (where path 1 i the complete branch 1 and path 2 is the first 2 relations in branch 2) mentioned before the question, and ask the model to only answer the question with the facts given in the path 1 and path 2.
Avoid using external knowledge or assumptions. Do not refer to the paths directly (e.g., avoid saying 'branch 1' or 'branch 2'). 

Return your output in JSON format like this:
{ "question": "<your framed question>" }
"""
    },
    {
        "role": "system", 
        "content": "{'question': 'Consider the two paths given below and answer the following question using only the options and no explanations.\n\n- Path 1:\nGreta Thunberg led a school strike.\nThe school strike gained global media attention.\nMedia attention influenced environmental policy.\nEnvironmental policy relates to climate change.\n\n- Path 2:\nGreta Thunberg gave a speech at the UN Youth Summit.\nThe summit was attended by António Guterres.\n\nQuestion: based on the given above information can we infer with complete confidence the fact(s) that:\nAntónio Guterres emphasized youth involvement in global leadership.\n Youth involvement relates to climate change. \n\nAnswer using only one of the following options:\nTRUE\nFALSE\nCannot conclude from the given context.\n Selecting True/False means that there is enough information from path 1 and path 2, to conclude if the asked relations in the question are true or false.'}"
    },
    {        
        "role": "user",
        "content": f""" Simlar to the above example, you are given a diamond pattern in a knowledge graph: 
- Branch 1 (Background):
{branch1_text}
- Branch 2 (setup for Question):
{branch2_text}

Now, write a True/False/Cannot_infer-style question based on this setup. The question should be made by giving the entire first path, and then first 2 relations of the second path as to be considered as complete knowledge, and ask a question by giving remaining relations of the second branch and asking if the model is able to tell if the remaining part given is true based on the given branch 1 and partial branch 2 or if its false, or if there is no sufficient knowledge given based on which we can infer. Always end your question in these two lines: \n\nAnswer using only one of the following options:\nTRUE\nFALSE\nCannot conclude from the given context.\n Selecting True/False means that there is enough information from path 1 and path 2, to conclude if the asked relations in the question are true or false.
Also make sure that the prompt has the senetence 'paqth 1' and 'path 2' (where path 1 i the complete branch 1 and path 2 is the first 2 relations in branch 2) mentioned before the question, and ask the model to only answer the question with the facts given in the path 1 and path 2.
Avoid using external knowledge or assumptions. Do not refer to the paths directly (e.g., avoid saying 'branch 1' or 'branch 2'). 

Return your output in JSON format containing the framed question only. make sure the output is in the form of {{'question': '<your framed question>'}} that is only use single quotes for the question, and to enclose the your framed question without any newline characters in the starting or ending it could be there anywhere in between."""
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

# print("Generating response...")
# output = llama(sample_prompt, max_new_tokens=400)
# print("Generated Response:")
# print(output[0]["generated_text"][-1])

# raw = output[0]["generated_text"][-1]['content']

# print("Raw Response:")
# print(raw)
# print(type(raw))

# Extract everything between 'question': '...' and write the extracted question into a prompt.json file
# match = re.search(r"'question':\s*'(.*?)'}", raw, re.DOTALL)
# if match:
#     question = match.group(1)
#     print("Extracted Question:\n")
#     print(question)
#     # Write the extracted question into a prompt.json file
#     with open("prompt.json", "w") as f:
#         json.dump({"prompt": question}, f, indent=4)
# else:
#     print("Could not extract question.")

all_prompts = []
# Loop through all diamonds in the dictionary and generate prompts
for diamond_key, diamond in diamonds_dict.items():
    print(f"Processing {diamond_key}...")
    sample_prompt = format_for_llama_prompt(diamond)

    print("Generating response...")
    output = llama(sample_prompt, max_new_tokens=400)

    raw = output[0]["generated_text"][-1]['content']


    # Extract everything between 'question': '...' and write the extracted question into a prompt.json file
    match = re.search(r"'question':\s*'(.*?)'}", raw, re.DOTALL)
    if match:
        question = match.group(1)
        all_prompts.append({"diamond_key": diamond_key, "prompt": question})
        print(f"Extracted Question for {diamond_key}:\n")
    else:
        print(f"Could not extract question for {diamond_key}.")

# Write all prompts to a prompts.json file
with open(f"prompts_{args.start_index}_{args.end_index}.json", "w") as f:
    json.dump({"prompts": all_prompts}, f, indent=4)

print(f"Wrote {len(all_prompts)} prompts to prompts_{args.start_index}_{args.end_index}.json")
