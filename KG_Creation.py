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
        current_step_size = 64
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
            
        return outputs
    


    def parse_entities(self, document: str) -> list:
        """Parses the entities in the document inputted using the LLM"""

        prompt = """
You are a political expert. Considering the following passage, extract all important entities inside of it, before returning them in a numbered list.
The list should place each entity on a separate line, such as:
1. Barack Obama
2. White House
...

Passage: """ + document + "\n Now provide the numbered list of entities: "
        
        entity_string = self.run_model([prompt])[0]

        pattern = r'^\d+\.\s*(.+)$'
        matches = re.findall(pattern, entity_string, re.MULTILINE)
        return matches
    


    def parse_relations(self, entities: list, document: str):
        """Parses the relations between every pair of entities based on the document using the LLM"""

        prompt = f"""
You are a political expert. Consider the following passage. Then consider the two entities that were extracted from this passage.
Place the single relation between the two entities on a separate line starting with 'RELATION: '.

For example, we could have:
Entity 1: Barack Obama
Entity 2: White House
RELATION: lives in

Passage: {document}

"""
        prompts = []
        entity_relations = []

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                formatted_prompt = prompt + f"""Entity 1: {entities[i]}
Entity 2: {entities[j]}"""
                prompts.append((entities[i], entities[j], formatted_prompt))

        outputs = self.run_model([p[2] for p in prompts])
        for output_idx, output in enumerate(outputs):
            match = re.search(r'^RELATION:\s*(.+)', output, re.MULTILINE)
            entity_relations.append((prompts[output_idx][0], prompts[output_idx][1], match.group(1)))
        
        return entity_relations



    def create_kg(self):
        """Creates KG from documents"""

        if self.data == None:
            self.parse_data()
        
        for document in self.data:
            entities = self.parse_entities(document)
            print("Number of Entities:", len(entities))
            entity_relations = self.parse_relations(entities, document)

            for ent1, ent2, relation in entity_relations:
                self.G.add_edge(ent1, ent2, relation = relation)

        print("Number of documents:", len(self.data))
    

    def save_kg(self):
        """Save the KG"""
        nx.write_edgelist(self.G, "graph_edgelist", data = True)

if __name__ == "__main__":
    start = time.time()
    graph_creator = GraphCreator(None)
    graph_creator.parse_data()
    graph_creator.create_kg()

    graph_creator.save_kg()

    print("Total Time:", time.time() - start)


 