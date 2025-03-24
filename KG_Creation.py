""" Creates a Knowledge Graph from a directory of input documents. """

import os
os.environ['HF_HOME'] = '/scratch/drjieliu_root/drjieliu0/lxguan/huggingface_cache'#'/scratch/drjieliu_root/drjieliu/lxguan/huggingface_cache'#
os.environ['TRANSFORMERS_CACHE'] = '/scratch/drjieliu_root/drjieliu0/lxguan/huggingface_cache'#'/scratch/drjieliu_root/drjieliu/lxguan/huggingface_cache'#

import numpy as np
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import torch
import time
torch.manual_seed(42)


class GraphCreator:
    def __init__(self, data = None):
        self.G = nx.DiGraph()
        self.model = AutoModelForCausalLM.from_pretrained("/nfs/turbo/umms-drjieliu/proj/llm_kg/Llama-3.1-8B-Instruct", torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained("/nfs/turbo/umms-drjieliu/proj/llm_kg/Llama-3.1-8B-Instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data = data



    def parse_data(self) -> list:
        """Loads data from disk into a list of text chunks"""
        text_chunks = []
        directory = "data"
        # Iterate through all files in the directory
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)

            # Ensure it's a text file
            if os.path.isfile(file_path) and file_name.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

                # Tokenize and chunk the text into chunks of at most 600 tokens
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                for i in range(0, len(tokens), 600):
                    chunk = tokens[i:i+600]
                    text_chunk = self.tokenizer.decode(chunk, clean_up_tokenization_spaces=True)
                    text_chunks.append(text_chunk)

        # Save the list of text chunks as self.data
        self.data = text_chunks



    def run_model(self, prompts: list[str]) -> str:
        """Runs the model with the input prompt."""

        outputs = []
        current_step_size = 16
        idx = 0
        with torch.inference_mode():
            while idx < len(prompts):
                batch_prompts = prompts[idx: idx + current_step_size]
                
                # Create message format
                messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
                
                try:
                    # Clear CUDA cache before processing
                    torch.cuda.empty_cache()
                    
                    # Apply chat template
                    templated_prompts = self.tokenizer.apply_chat_template(
                        conversation=messages,
                        tokenize=False,
                        return_tensors="pt"
                    )
                    
                    
                    # Tokenize
                    tokens = self.tokenizer(
                        templated_prompts,
                        padding=True,
                        add_special_tokens=False,
                        return_tensors="pt"
                    )
                    # Move to GPU and generate
                    tokens = {k: v.to("cuda") for k, v in tokens.items()}
                    generated = self.model.generate(
                        **tokens,
                        do_sample=False,
                        top_p=0.95,
                        max_new_tokens=2048,
                    )
                    
                    # Process outputs
                    batch_outputs = self.tokenizer.batch_decode(
                        [out[inp.shape[0]:] for out, inp in zip(generated, tokens['input_ids'])],
                        skip_special_tokens=True
                    )
                    
                    outputs.extend(batch_outputs)
                    
                    # print(f"Successfully processed batch with size {current_step_size}")
                    idx += current_step_size  # Move to next batch
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Clear CUDA cache
                        torch.cuda.empty_cache()
                        
                        # Reduce batch size
                        new_step_size = max(1, current_step_size // 2)
                        print(f"OOM error encountered. Reducing batch size from {current_step_size} to {new_step_size}")
                        
                        if new_step_size == current_step_size:
                            print("Cannot reduce batch size further. Already at minimum.")
                            print(batch_prompts, flush=True)
                            raise
                        
                        current_step_size = new_step_size
                        # Don't increment idx - retry with same batch but smaller size
                    else:
                        print(f"Non-OOM error in batch: {str(e)}")
                        raise
                except Exception as e:
                    print(f"Error in batch: {str(e)}")
                    raise
            
        # print("Generation Time:", time.time() - start, flush=True)
        return outputs
    
    

    def parse_document(self, documents):
        """Parses documents using a variant of the Graph RAG prompt and extracts entities and relations. """

        prompt = """
-Goal-
Given a text document, identify all entities from the text and all relationships among the identified entities.
 
-Steps-
1. Identify all entities.
 
2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, consider the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: how the source entity and the target entity are related to each other
 
3. Return output in English as a numbered list of all the entities and relationships in triples identified in steps 1 and 2.

Note: Our main goal is that final numbered list of 3-tuples. Under no circumstances should you produce code, an accompanying explanation, etc. Just follow the above steps.
 
######################
-Examples-
######################
Example 1:
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
List: 
1. (Martin Smith, Central Institution, is the chair of)

######################
-Real Data-
######################
Text: """
        
        prompts = []

        for document in documents:
            prompts.append(prompt + document + "\n Output: ")

        document_outputs = self.run_model(prompts)

        # Regex to capture triples in the format ("source_entity", "target_entity", "relationship_description")
        pattern = r"\d+\.\s\(([^,]+),\s([^,]+),\s([^)]+)\)"

        outputs = []
        for output in document_outputs:

            matches = re.findall(pattern, output)
            outputs.append(matches)
        return outputs
        






    def create_kg(self):
        """Creates KG from documents"""

        if self.data == None:
            self.parse_data()
        
        all_entities = self.parse_document(self.data[:8])

        for entities_and_relations in all_entities:
            for ent1, ent2, relation in entities_and_relations:
                self.G.add_edge(ent1, ent2, relation = relation)

        print("Number of documents:", len(self.data))
    

    def save_kg(self):
        """Save the KG"""
        nx.write_edgelist(self.G, "graph_edgelist", data = True, delimiter="|")

if __name__ == "__main__":
    start = time.time()
    graph_creator = GraphCreator(None)
    graph_creator.parse_data()
    graph_creator.create_kg()

    graph_creator.save_kg()
