# Runs Llama-2-7B-chat-hf on the evaluation set

import os
os.environ['HF_HOME'] = '/scratch/drjieliu_root/drjieliu/lxguan/huggingface_cache'#'/scratch/drjieliu_root/drjieliu0/lxguan/huggingface_cache'#
os.environ['TRANSFORMERS_CACHE'] = '/scratch/drjieliu_root/drjieliu/lxguan/huggingface_cache'#'/scratch/drjieliu_root/drjieliu0/lxguan/huggingface_cache'#
import numpy as np
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
import re
import torch
import argparse
import json
import time
torch.manual_seed(42)

class Evaluation:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        pass

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
        if isinstance(outputs, list):
            return outputs[0]
        return outputs

    def get_next_token_logits(self, prompt):
        """
        Get logits for the next token after the prompt.
        
        Args:
            prompt: Either a single string or a list of strings
            
        Returns:
            Tensor of shape [batch_size, vocab_size]
        """
        is_single = isinstance(prompt, str)
        prompts = [prompt] if is_single else prompt
        
        outputs = []
        initial_step_size = 64
        current_step_size = initial_step_size
        idx = 0
        
        while idx < len(prompts):
            batch_prompts = prompts[idx:idx + current_step_size]
            
            try:
                # Clear CUDA cache before processing batch
                torch.cuda.empty_cache()
                
                # Create message format and apply template for current batch
                messages = [{"role": "user", "content": p} for p in batch_prompts]
                templated_prompts = [
                    self.tokenizer.apply_chat_template(
                        conversation=[msg],
                        tokenize=False,
                        return_tensors="pt"
                    ) for msg in messages
                ]
                
                # Tokenize full batch
                tokens = self.tokenizer(
                    templated_prompts,
                    padding=True,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to("cuda")
                
                # Process full batch at once
                with torch.no_grad():
                    outputs_batch = self.model(**tokens)
                    batch_logits = outputs_batch.logits[:, -1, :].cpu()  # Move to CPU immediately
                    outputs.append(batch_logits)
                
                # Clear batch tensors
                del tokens
                del outputs_batch
                torch.cuda.empty_cache()
                
                idx += current_step_size
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    new_step_size = max(1, current_step_size // 2)
                    print(f"OOM error encountered. Reducing batch size from {current_step_size} to {new_step_size}")
                    
                    if new_step_size == current_step_size:
                        print("Cannot reduce batch size further. Already at minimum.")
                        raise
                    
                    current_step_size = new_step_size
                else:
                    print(f"Non-OOM error in batch: {str(e)}")
                    raise
            except Exception as e:
                print(f"Error in batch: {str(e)}")
                raise
        
        # Concatenate all batch outputs
        if len(outputs) > 1:
            final_logits = torch.cat(outputs, dim=0)
        else:
            final_logits = outputs[0]
        
        return final_logits.to("cuda")  # Move back to GPU for final operations

    def process_logit_outputs(self, llm_response):

        true_id = self.tokenizer.encode("True")[1]
        false_id = self.tokenizer.encode("False")[1]
        none_id = self.tokenizer.encode("None")[1]

        # Extract the logits for just these three tokens
        selected_logits = llm_response[0, [true_id, false_id, none_id]]

        # Apply softmax only to these three logits
        selected_probs = torch.softmax(selected_logits, dim=-1)

        # Extract individual probabilities
        yes_prob = selected_probs[0].item()  # True
        no_prob = selected_probs[1].item()   # False
        none_prob = selected_probs[2].item() # None
        
        # Need to softmax just the 3...
        triple = (yes_prob, no_prob, none_prob)
        return ["True", "False", "None"][np.argmax(triple)], triple

    
    def load_files(self, base_folder = "prompts"):
        
        file_arr = []
        for filename in os.listdir(base_folder):
            if filename.endswith(".json"):
                file_path = os.path.join(base_folder, filename)
                with open(file_path, "rb") as f:
                    data = json.load(f)
                    file_arr.append((filename, data))
        
        return file_arr
    
    def run_prompts(self):
        file_arr = self.load_files()
        
        for filename, json_file in file_arr:
            subset_arr = []
            for prompt_object in json_file['prompts']:
                prompt_appendage = " Return either 'True', 'False', or 'None' directly as your outputs, with None corresponding to if there is insufficient information."
                llm_response = self.get_next_token_logits(prompt_object['prompt'] + prompt_appendage)
                response, triple = self.process_logit_outputs(llm_response)
                prompt_object['output'] = response
                prompt_object['output_probs'] = list(triple)
                subset_arr.append(prompt_object)
            with open("outputs/output" + filename[7:], "w") as f:
                json.dump({"prompts": subset_arr}, f, indent=4)

    def get_explanations(self, file_idx):
        file_arr = self.load_files("prompts_new")
        
        file_arr = [sorted(file_arr)[file_idx]]
        
        for filename, json_file in file_arr:
            subset_arr = []
            for local_idx, prompt_object in enumerate(json_file['prompts']):
                prompt_appendage = ""#" Return either 'TRUE' or 'FALSE', with FALSE corresponding to if you have ANY suspicion that there is insufficient information. Return an explanation along with your choice. "
                output = self.run_model([prompt_object['prompt'] + prompt_appendage])
                prompt_object["explanation"] = output
                subset_arr.append(prompt_object)
                if local_idx % 100 == 0:
                    print(output, flush=True)
                    print("\n\n\n")
            with open("outputs_new/output" + filename[7:], "w") as f:
                json.dump({"prompts": subset_arr}, f, indent=4)



if __name__ == "__main__":
    model_path = "/scratch/drjieliu_root/drjieliu/lxguan/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_idx", type=int, help="Which file to use, 0-5")
    args = parser.parse_args()

    eval_obj = Evaluation(tokenizer, model)
    eval_obj.get_explanations(args.file_idx)