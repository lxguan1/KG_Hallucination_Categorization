""" Creates a Knowledge Graph from a directory of input documents. """

import os
import numpy as np
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
import re
import torch
import argparse
import time
torch.manual_seed(42)


class GraphCreator:
    def __init__(self, data = None):
        self.G = nx.DiGraph()
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data = data



    def parse_data(self, args) -> list:
        """Loads data from disk into a list of text chunks"""
        text_chunks = []
        directory = "Data/hearings txt"
        node_parser = SentenceSplitter(chunk_size=600, chunk_overlap=0)

        text_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.txt')]
        
        text_files.sort()

        selected_files = text_files[args.start_document:args.end_document]
        # Iterate through all files in the directory
        for file_name in selected_files:
            file_path = os.path.join(directory, file_name)

            # Ensure it's a text file
            if os.path.isfile(file_path) and file_name.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

                # Use LlamaIndex SentenceSplitter to split the text
                nodes = node_parser.get_nodes_from_documents([Document(text=text)])
                
                for node in nodes:
                    text_chunks.append(node.text)
        print("Text Chunks:", len(text_chunks), flush=True)
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
 
3. Return output in English as a numbered list of all the entities and relationships in |-separated triples identified in steps 1 and 2.

Note: Our main goal is that final numbered list of 3-tuples. Under no circumstances should you produce code, an accompanying explanation, etc. Just follow the above steps.
Note: the triples (Entity1|Entity2|Relation) should be Entity1 has relation Relation with Entity2. Remember to put the parentheses.
 
######################
-Examples-
######################
Example 1:
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
List: 
1. (Martin Smith|Central Institution|is the chair of)
2. (Central Institution|Monday|is scheduled to meet)
3. (Central Institution|Thursday|is scheduled to meet)

######################
-Real Data-
######################
Text: """
        
        prompts = []

        for document in documents:
            prompts.append(prompt + document + "\n Output: ")

        document_outputs = self.run_model(prompts)

        # Regex to capture triples in the format ("source_entity", "target_entity", "relationship_description")
        pattern = r"\d+\.\s*\(([^|]+)\|([^|]+)\|([^)]+)\)"

        outputs = []
        for output in document_outputs:

            matches = re.findall(pattern, output)
            # Clean each triple element
            cleaned_matches = []
            for match in matches:
                # Strip whitespace and ensure no numbered list items are accidentally included
                src = match[0].strip()
                tgt = match[1].strip()
                rel = match[2].strip()
                
                # Remove any explanatory text after a dash
                src = re.sub(r'\s*-\s*.*$', '', src)
                tgt = re.sub(r'\s*-\s*.*$', '', tgt)
                rel = re.sub(r'\s*-\s*.*$', '', rel)
                
                # Remove any text after a closing parenthesis
                src = re.sub(r'\)\s*.*$', '', src)
                tgt = re.sub(r'\)\s*.*$', '', tgt)
                rel = re.sub(r'\)\s*.*$', '', rel)
                
                # Remove any trailing parentheses that might be left
                src = re.sub(r'\)\s*$', '', src)
                tgt = re.sub(r'\)\s*$', '', tgt)
                rel = re.sub(r'\)\s*$', '', rel)
                
                # Remove any trailing pipes or commas
                src = re.sub(r'[|,]\s*$', '', src)
                tgt = re.sub(r'[|,]\s*$', '', tgt)
                rel = re.sub(r'[|,]\s*$', '', rel)
                
                # Replace any remaining pipes within the entities or relation with spaces or dashes
                src = src.replace('|', ' ')
                tgt = tgt.replace('|', ' ')
                rel = rel.replace('|', ' ')

                if src and tgt and rel:
                    cleaned_matches.append((src, tgt, rel))
            outputs.append(cleaned_matches)

        # Subsequent passes

        gleaning_prompt = """-Goal-
Given a text document, identify all entities from the text and all relationships among the identified entities.
 
-Steps-
1. Identify all entities in the text.
 
2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, consider the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: how the source entity and the target entity are related to each other
Do not include relations that are already extracted.
 
3. Return output in English as a numbered list of all the entities and relationships in |-separated triples identified in steps 1 and 2.

Note: Our main goal is that final numbered list of 3-tuples. Under no circumstances should you produce code, an accompanying explanation, etc. Just follow the above steps.
Note: If no entities and relationships exist that haven't been extracted in the "Already Extracted" list, return "No new relations exist". 
Note: the triples (Entity1|Entity2|Relation) should be Entity1 has relation Relation with Entity2. Remember to put the parentheses.
######################
-Examples-
######################
Example 1:
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
Already Extracted:
[]
######################
Output:
List: 
1. (Martin Smith|Central Institution|is the chair of)
2. (Central Institution|Monday|is scheduled to meet)
3. (Central Institution|Thursday|is scheduled to meet)

######################
-Real Data-
######################
Text: """
        for _ in range(2):
            # Glean 2 times
            gleaning_prompts = []
            for document, output in zip(documents, outputs):
                extracted_str = str([f"({e1}|{e2}|{r})" for e1, e2, r in output])
                gleaning_prompts.append(gleaning_prompt + document + "\n Already Extracted:\n " + extracted_str + "\n Output: ")

            document_outputs = self.run_model(gleaning_prompts)
            gleaning_outputs = []
            for output in document_outputs:

                matches = re.findall(pattern, output)
                # Clean each triple element
                cleaned_matches = []
                for match in matches:
                    # Strip whitespace and ensure no numbered list items are accidentally included
                    src = match[0].strip()
                    tgt = match[1].strip()
                    rel = match[2].strip()
                    
                    # Remove any explanatory text after a dash
                    src = re.sub(r'\s*-\s*.*$', '', src)
                    tgt = re.sub(r'\s*-\s*.*$', '', tgt)
                    rel = re.sub(r'\s*-\s*.*$', '', rel)
                    
                    # Remove any text after a closing parenthesis
                    src = re.sub(r'\)\s*.*$', '', src)
                    tgt = re.sub(r'\)\s*.*$', '', tgt)
                    rel = re.sub(r'\)\s*.*$', '', rel)
                    
                    # Remove any trailing parentheses that might be left
                    src = re.sub(r'\)\s*$', '', src)
                    tgt = re.sub(r'\)\s*$', '', tgt)
                    rel = re.sub(r'\)\s*$', '', rel)
                    
                    # Remove any trailing pipes or commas
                    src = re.sub(r'[|,]\s*$', '', src)
                    tgt = re.sub(r'[|,]\s*$', '', tgt)
                    rel = re.sub(r'[|,]\s*$', '', rel)
                    
                    # Replace any remaining pipes within the entities or relation with spaces or dashes
                    src = src.replace('|', ' ')
                    tgt = tgt.replace('|', ' ')
                    rel = rel.replace('|', ' ')

                    if src and tgt and rel:
                        cleaned_matches.append((src, tgt, rel))
                gleaning_outputs.append(cleaned_matches)
            outputs = [list(set(output + gleaning_output)) for output, gleaning_output in zip(outputs, gleaning_outputs)]
        
        return outputs
        






    def create_kg(self):
        """Creates KG from documents"""

        if self.data == None:
            self.parse_data()
        
        all_entities = self.parse_document(self.data)

        for entities_and_relations in all_entities:
            for ent1, ent2, relation in entities_and_relations:
                self.G.add_edge(ent1, ent2, relation = relation)

        print("Number of documents:", len(self.data))
    

    def save_kg(self, args):
        """Save the KG"""
        nx.write_edgelist(self.G, "graph_edgelist_" + str(args.start_document), data = True, delimiter="|")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_document", default=0, type=int, help= "Which document in the data directory to start at?")
    parser.add_argument("--end_document", default=30, type=int, help= "Which document in the data directory to end at?")
    args = parser.parse_args()
    start = time.time()
    graph_creator = GraphCreator(None)
    graph_creator.parse_data(args)
    graph_creator.create_kg()

    graph_creator.save_kg(args)
