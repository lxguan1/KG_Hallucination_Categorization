""" Creates a Knowledge Graph from a directory of input documents. """

import os
os.environ['HF_HOME'] = '/scratch/drjieliu_root/drjieliu/lxguan/huggingface_cache'#'/scratch/drjieliu_root/drjieliu0/lxguan/huggingface_cache'#
os.environ['TRANSFORMERS_CACHE'] = '/scratch/drjieliu_root/drjieliu/lxguan/huggingface_cache'#'/scratch/drjieliu_root/drjieliu0/lxguan/huggingface_cache'

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
        self.model = AutoModelForCausalLM.from_pretrained("/scratch/drjieliu_root/drjieliu/lxguan/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained("/scratch/drjieliu_root/drjieliu/lxguan/Llama-2-7b-chat-hf")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data = data



    def parse_data(self, args) -> list:
        """Loads data from disk into a list of text chunks"""
        text_chunks = []
        directory = "data"
        node_parser = SentenceSplitter(chunk_size=600, chunk_overlap=0)
        # Iterate through all files in the directory
        for file_name in os.listdir(directory):
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
        self.data = text_chunks[args.start_document:args.end_document]



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
    
    

    def parse_document(self, documents, num_gleanings=2, analyze_gleaning=True):
        """
        Parses documents using a variant of the Graph RAG prompt and extracts entities and relations.
        
        Parameters:
        -----------
        documents : list
            List of document text chunks to analyze
        num_gleanings : int
            Number of additional gleaning passes to perform
        analyze_gleaning : bool
            Whether to analyze and report on the effectiveness of gleaning passes
            
        Returns:
        --------
        outputs : list
            List of extracted entity-relation triples
        """

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

        initial_outputs = []
        for output in document_outputs:
            matches = re.findall(pattern, output)
            initial_outputs.append(matches)

        # Store the outputs from each gleaning step for analysis
        outputs = initial_outputs
        gleaning_stats = {
            'initial_count': [len(triples) for triples in outputs],
            'gleaning_counts': [],
            'unique_triples_added': [],
            'percent_increase': []
        }

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
 
3. Return output in English as a numbered list of all the entities and relationships in triples identified in steps 1 and 2.

Note: Our main goal is that final numbered list of 3-tuples. Under no circumstances should you produce code, an accompanying explanation, etc. Just follow the above steps.
Note: If no entities and relationships exist that haven't been extracted in the "Already Extracted" list, return "No new relations exist". 
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
1. (Martin Smith, Central Institution, is the chair of)

######################
-Real Data-
######################
Text: """

        # Track total triples found at each step
        cumulative_triples = []
        for doc_triples in outputs:
            cumulative_triples.append(set([(e1, e2, r) for e1, e2, r in doc_triples]))

        for gleaning_step in range(num_gleanings):
            # Glean multiple times
            gleaning_prompts = []
            for document, output in zip(documents, outputs):
                # Create a string representation of the already extracted triples
                extracted_str = str([f"({e1}, {e2}, {r})" for e1, e2, r in output])
                gleaning_prompts.append(gleaning_prompt + document + "\n Already Extracted:\n " + extracted_str + "\n Output: ")

            document_outputs = self.run_model(gleaning_prompts)
            print(document_outputs[0])
            gleaning_outputs = []
            
            step_new_triples = []
            for output in document_outputs:
                matches = re.findall(pattern, output)
                gleaning_outputs.append(matches)
                step_new_triples.append(len(matches))
            
            # Record statistics before merging
            pre_merge_counts = [len(doc_triples) for doc_triples in outputs]
            
            # For each document, track how many new unique triples were added
            new_unique_counts = []
            for i, (current_triples, new_triples) in enumerate(zip(outputs, gleaning_outputs)):
                current_set = set([(e1, e2, r) for e1, e2, r in current_triples])
                new_set = set([(e1, e2, r) for e1, e2, r in new_triples])
                unique_new = new_set - current_set
                new_unique_counts.append(len(unique_new))
                
                # Add to cumulative set
                cumulative_triples[i].update(unique_new)
            
            # Merge the outputs with existing triples
            outputs = [sorted(list(set(output + gleaning_output))) for output, gleaning_output in zip(outputs, gleaning_outputs)]
            post_merge_counts = [len(doc_triples) for doc_triples in outputs]
            
            # Calculate percentage increase
            percent_increases = []
            for pre, post in zip(pre_merge_counts, post_merge_counts):
                if pre > 0:
                    percent_increases.append((post - pre) / pre * 100)
                else:
                    percent_increases.append(float('inf') if post > 0 else 0)
            
            # Store statistics for this gleaning step
            gleaning_stats['gleaning_counts'].append(step_new_triples)
            gleaning_stats['unique_triples_added'].append(new_unique_counts)
            gleaning_stats['percent_increase'].append(percent_increases)
        
        if analyze_gleaning:
            self._analyze_gleaning_effectiveness(gleaning_stats, num_gleanings)
        
        return outputs
    
    def _analyze_gleaning_effectiveness(self, stats, num_gleanings):
        """Analyze and print statistics about the effectiveness of gleaning passes"""
        
        # Calculate total triples found at each stage
        total_initial = sum(stats['initial_count'])
        total_after_each_gleaning = [total_initial]
        
        for i in range(num_gleanings):
            added_this_step = sum(stats['unique_triples_added'][i])
            total_after_each_gleaning.append(total_after_each_gleaning[-1] + added_this_step)
        
        # Print summary statistics
        print("\n=== GLEANING EFFECTIVENESS ANALYSIS ===")
        print(f"Number of documents analyzed: {len(stats['initial_count'])}")
        print(f"Initial extraction: {total_initial} triples ({total_initial/len(stats['initial_count']):.2f} per document)")
        
        print("\nGleaning Results:")
        print("| Pass | New Triples | Cumulative Total | % Increase | % Increase from Initial |")
        print("|------|-------------|------------------|------------|-------------------------|")
        
        for i in range(num_gleanings):
            new_triples = sum(stats['unique_triples_added'][i])
            cumulative = total_after_each_gleaning[i+1]
            
            # Calculate percentage increases
            increase_from_previous = (new_triples / total_after_each_gleaning[i] * 100) if total_after_each_gleaning[i] > 0 else float('inf')
            increase_from_initial = ((cumulative - total_initial) / total_initial * 100) if total_initial > 0 else float('inf')
            
            print(f"| {i+1}    | {new_triples:11d} | {cumulative:16d} | {increase_from_previous:8.2f}% | {increase_from_initial:23.2f}% |")
        
        # Analyze diminishing returns
        if num_gleanings >= 2:
            print("\nDiminishing Returns Analysis:")
            
            # Calculate the rate of decrease in new triples
            decrease_rates = []
            for i in range(1, num_gleanings):
                previous = sum(stats['unique_triples_added'][i-1])
                current = sum(stats['unique_triples_added'][i])
                
                if previous > 0:
                    decrease_rate = (previous - current) / previous * 100
                    decrease_rates.append(decrease_rate)
                    print(f"Gleaning {i+1} yielded {decrease_rate:.2f}% fewer new triples than gleaning {i}")
            
            # Recommend optimal number of gleanings
            if decrease_rates:
                avg_decrease_rate = sum(decrease_rates) / len(decrease_rates)
                
                # Look for the point where decrease rate exceeds a threshold (e.g., 50%)
                diminishing_threshold = 50
                optimal_gleanings = num_gleanings  # Default to current number
                
                for i, rate in enumerate(decrease_rates):
                    if rate > diminishing_threshold:
                        optimal_gleanings = i + 1  # +1 because we're looking at gleaning i+2 vs i+1
                        break
                
                if optimal_gleanings < num_gleanings:
                    print(f"\nRECOMMENDATION: Consider using {optimal_gleanings} gleaning passes, as further passes show diminishing returns.")
                else:
                    print(f"\nRECOMMENDATION: Current {num_gleanings} gleaning passes are effective. Consider testing with more passes to find optimal point.")
            
            # Calculate efficiency (triples per gleaning)
            efficiency = []
            for i in range(num_gleanings):
                new_triples = sum(stats['unique_triples_added'][i])
                efficiency.append(new_triples)
            
            print("\nEfficiency (new triples per gleaning):")
            for i, eff in enumerate(efficiency):
                print(f"Gleaning {i+1}: {eff} new triples ({eff/len(stats['initial_count']):.2f} per document)")
        
        print("\n=== END OF ANALYSIS ===\n")
        






    def create_kg(self):
        """Creates KG from documents"""

        if self.data == None:
            self.parse_data()
        
        all_entities = self.parse_document(self.data)

        for entities_and_relations in all_entities:
            for ent1, ent2, relation in entities_and_relations:
                self.G.add_edge(ent1, ent2, relation = relation)

        print("Number of documents:", len(self.data))
    

    def save_kg(self):
        """Save the KG"""
        nx.write_edgelist(self.G, "graph_edgelist", data = True, delimiter="|")

def analyze_graph(G, title="Graph Analysis Report", save_report=True, output_file="graph_report.txt", 
                  include_visualizations=True, vis_output_dir="./graph_visualizations/"):
    """
    Generate a comprehensive analysis report of a networkx graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to analyze
    title : str
        Title of the report
    save_report : bool
        Whether to save the report to a file
    output_file : str
        Path to save the report
    include_visualizations : bool
        Whether to generate and save visualizations
    vis_output_dir : str
        Directory to save visualizations
        
    Returns:
    --------
    report : str
        The text report with all analysis
    """
    # Create report sections
    sections = []
    
    # Basic Information
    sections.append(f"# {title}\n")
    sections.append("## 1. Basic Graph Information\n")
    
    is_directed = G.is_directed()
    is_weighted = is_weighted_graph(G)
    basic_info = [
        f"Graph Type: {'Directed' if is_directed else 'Undirected'} {'Weighted' if is_weighted else 'Unweighted'} Graph",
        f"Number of Nodes: {G.number_of_nodes()}",
        f"Number of Edges: {G.number_of_edges()}",
        f"Graph Density: {nx.density(G):.6f}",
    ]
    
    # Add node and edge attribute info
    if G.nodes and len(G.nodes) > 0:
        first_node = list(G.nodes)[0]
        if G.nodes[first_node]:
            node_attrs = list(G.nodes[first_node].keys())
            basic_info.append(f"Node Attributes: {', '.join(node_attrs)}")
    
    if G.edges and len(G.edges) > 0:
        first_edge = list(G.edges)[0]
        if G.edges[first_edge]:
            edge_attrs = list(G.edges[first_edge].keys())
            basic_info.append(f"Edge Attributes: {', '.join(edge_attrs)}")
    
    sections.append("\n".join(basic_info) + "\n")
    
    # Connectivity Analysis
    sections.append("\n## 2. Connectivity Analysis\n")
    connectivity_info = []
    
    # Is the graph connected?
    if is_directed:
        is_connected = nx.is_strongly_connected(G)
        connectivity_info.append(f"Strongly Connected: {is_connected}")
        
        if not is_connected:
            strongly_connected_components = list(nx.strongly_connected_components(G))
            connectivity_info.append(f"Number of Strongly Connected Components: {len(strongly_connected_components)}")
            connectivity_info.append(f"Largest Strongly Connected Component Size: {len(max(strongly_connected_components, key=len))}")
            
        is_weakly_connected = nx.is_weakly_connected(G)
        connectivity_info.append(f"Weakly Connected: {is_weakly_connected}")
        
        if not is_weakly_connected:
            weakly_connected_components = list(nx.weakly_connected_components(G))
            connectivity_info.append(f"Number of Weakly Connected Components: {len(weakly_connected_components)}")
    else:
        is_connected = nx.is_connected(G)
        connectivity_info.append(f"Connected: {is_connected}")
        
        if not is_connected:
            connected_components = list(nx.connected_components(G))
            connectivity_info.append(f"Number of Connected Components: {len(connected_components)}")
            connectivity_info.append(f"Largest Connected Component Size: {len(max(connected_components, key=len))}")
    
    # Try to compute average shortest path length
    try:
        if is_connected or (is_directed and is_weakly_connected):
            avg_path = nx.average_shortest_path_length(G)
            connectivity_info.append(f"Average Shortest Path Length: {avg_path:.4f}")
        else:
            # For disconnected graphs, compute for the largest component
            if is_directed:
                largest_cc = max(nx.weakly_connected_components(G), key=len)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
            largest_subgraph = G.subgraph(largest_cc)
            avg_path = nx.average_shortest_path_length(largest_subgraph)
            connectivity_info.append(f"Average Shortest Path Length (largest component): {avg_path:.4f}")
    except Exception as e:
        connectivity_info.append(f"Could not compute Average Shortest Path Length: {str(e)}")
    
    # Calculate graph diameter (maximum shortest path)
    try:
        if is_connected or (is_directed and is_weakly_connected):
            diameter = nx.diameter(G)
            connectivity_info.append(f"Graph Diameter: {diameter}")
        else:
            # For disconnected graphs, compute for the largest component
            if is_directed:
                largest_cc = max(nx.weakly_connected_components(G), key=len)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
            largest_subgraph = G.subgraph(largest_cc)
            diameter = nx.diameter(largest_subgraph)
            connectivity_info.append(f"Graph Diameter (largest component): {diameter}")
    except Exception as e:
        connectivity_info.append(f"Could not compute Graph Diameter: {str(e)}")
    
    sections.append("\n".join(connectivity_info) + "\n")
    
    # Node Centrality Analysis
    sections.append("\n## 3. Node Centrality Analysis\n")
    
    # For large graphs, only analyze a subset of nodes
    node_sample = get_node_sample(G, max_nodes=10)
    
    # Calculate centrality measures
    try:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # For directed graphs, add in-degree and out-degree centrality
        if is_directed:
            in_degree_centrality = nx.in_degree_centrality(G)
            out_degree_centrality = nx.out_degree_centrality(G)
        
        # Eigenvector centrality (may not converge for some directed graphs)
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            eigenvector_centrality = None
        
        # Create a data frame for the top nodes by degree centrality
        centrality_data = []
        for node in node_sample:
            node_data = {
                'Node': node,
                'Degree Centrality': degree_centrality[node],
                'Betweenness Centrality': betweenness_centrality[node],
                'Closeness Centrality': closeness_centrality[node],
            }
            
            if is_directed:
                node_data['In-Degree Centrality'] = in_degree_centrality[node]
                node_data['Out-Degree Centrality'] = out_degree_centrality[node]
                
            if eigenvector_centrality:
                node_data['Eigenvector Centrality'] = eigenvector_centrality[node]
                
            centrality_data.append(node_data)
            
        # Convert to DataFrame and create a pretty table
        centrality_df = pd.DataFrame(centrality_data)
        centrality_table = tabulate(centrality_df, headers='keys', tablefmt='pipe', floatfmt='.4f')
        sections.append("Centrality measures for top nodes:\n")
        sections.append(centrality_table + "\n")
        
        # Add summary of centrality distributions
        sections.append("\nCentrality Distribution Summary:\n")
        centrality_summary = []
        
        for measure in ['Degree Centrality', 'Betweenness Centrality', 'Closeness Centrality']:
            if measure == 'Degree Centrality':
                values = list(degree_centrality.values())
            elif measure == 'Betweenness Centrality':
                values = list(betweenness_centrality.values())
            elif measure == 'Closeness Centrality':
                values = list(closeness_centrality.values())
                
            centrality_summary.append(f"{measure}:")
            centrality_summary.append(f"  - Mean: {np.mean(values):.4f}")
            centrality_summary.append(f"  - Median: {np.median(values):.4f}")
            centrality_summary.append(f"  - Max: {np.max(values):.4f} (Node: {list(degree_centrality.keys())[np.argmax(values)]})")
            centrality_summary.append(f"  - Min: {np.min(values):.4f}")
            centrality_summary.append(f"  - Standard Deviation: {np.std(values):.4f}")
            
        sections.append("\n".join(centrality_summary) + "\n")
    except Exception as e:
        sections.append(f"Could not compute centrality measures: {str(e)}\n")
    
    # Community Detection
    sections.append("\n## 4. Community Structure\n")
    
    try:
        # Try to detect communities using the Louvain method
        partition = community_louvain.best_partition(G.to_undirected() if is_directed else G)
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        
        sections.append(f"Number of communities detected: {len(communities)}\n")
        sections.append("Community size distribution:\n")
        
        community_sizes = [len(nodes) for nodes in communities.values()]
        community_size_summary = [
            f"  - Mean community size: {np.mean(community_sizes):.2f}",
            f"  - Median community size: {np.median(community_sizes):.2f}",
            f"  - Largest community size: {max(community_sizes)}",
            f"  - Smallest community size: {min(community_sizes)}"
        ]
        sections.append("\n".join(community_size_summary) + "\n")
        
        # Show top communities
        sections.append("\nLargest communities:\n")
        largest_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        
        for i, (comm_id, nodes) in enumerate(largest_communities):
            sample_nodes = nodes[:5]
            sections.append(f"Community {i+1} (ID: {comm_id}, Size: {len(nodes)})")
            sections.append(f"  Sample nodes: {', '.join(str(n) for n in sample_nodes)}")
            if len(nodes) > 5:
                sections.append(f"  ... and {len(nodes) - 5} more")
            sections.append("")
        
        # Calculate modularity
        try:
            modularity = community_louvain.modularity(partition, G.to_undirected() if is_directed else G)
            sections.append(f"Graph modularity: {modularity:.4f}\n")
        except Exception as e:
            sections.append(f"Could not compute modularity: {str(e)}\n")
            
    except Exception as e:
        sections.append(f"Could not perform community detection: {str(e)}\n")
    
    # Degree Distribution
    sections.append("\n## 5. Degree Distribution\n")
    
    if is_directed:
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        total_degrees = dict(G.degree())
        
        in_degree_counts = Counter(in_degrees.values())
        out_degree_counts = Counter(out_degrees.values())
        total_degree_counts = Counter(total_degrees.values())
        
        # In-degree stats
        in_degree_values = list(in_degrees.values())
        sections.append("In-degree distribution statistics:")
        sections.append(f"  - Mean in-degree: {np.mean(in_degree_values):.2f}")
        sections.append(f"  - Median in-degree: {np.median(in_degree_values):.2f}")
        sections.append(f"  - Max in-degree: {max(in_degree_values)} (Node: {list(in_degrees.keys())[np.argmax(list(in_degrees.values()))]})")
        sections.append(f"  - Min in-degree: {min(in_degree_values)}")
        sections.append(f"  - Standard deviation: {np.std(in_degree_values):.2f}")
        
        # Out-degree stats
        out_degree_values = list(out_degrees.values())
        sections.append("\nOut-degree distribution statistics:")
        sections.append(f"  - Mean out-degree: {np.mean(out_degree_values):.2f}")
        sections.append(f"  - Median out-degree: {np.median(out_degree_values):.2f}")
        sections.append(f"  - Max out-degree: {max(out_degree_values)} (Node: {list(out_degrees.keys())[np.argmax(list(out_degrees.values()))]})")
        sections.append(f"  - Min out-degree: {min(out_degree_values)}")
        sections.append(f"  - Standard deviation: {np.std(out_degree_values):.2f}")
    else:
        degrees = dict(G.degree())
        degree_counts = Counter(degrees.values())
        
        degree_values = list(degrees.values())
        sections.append("Degree distribution statistics:")
        sections.append(f"  - Mean degree: {np.mean(degree_values):.2f}")
        sections.append(f"  - Median degree: {np.median(degree_values):.2f}")
        sections.append(f"  - Max degree: {max(degree_values)} (Node: {list(degrees.keys())[np.argmax(list(degrees.values()))]})")
        sections.append(f"  - Min degree: {min(degree_values)}")
        sections.append(f"  - Standard deviation: {np.std(degree_values):.2f}")
    
    # Check for power-law distribution
    sections.append("\nPower-law distribution analysis:\n")
    try:
        if not is_directed:
            degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
        else:
            degree_sequence = sorted([d for n, d in G.in_degree()], reverse=True)
            
        # Fit power law
        if min(degree_sequence) > 0:  # Avoid log(0)
            log_degrees = np.log([x for x in degree_sequence if x > 0])
            log_counts = np.log([count for degree, count in Counter(degree_sequence).items() if degree > 0])
            
            if len(log_degrees) > 1 and len(log_counts) > 1:
                slope, intercept = np.polyfit(log_degrees, log_counts, 1)
                sections.append(f"Power-law exponent (α): {-slope:.4f}")
                
                if -3 <= slope <= -2:
                    sections.append("This is consistent with a scale-free network topology.")
                else:
                    sections.append("This does not strongly indicate a scale-free network topology.")
            else:
                sections.append("Not enough distinct degree values to fit power law.")
        else:
            sections.append("Cannot fit power law due to zero-degree nodes.")
    except Exception as e:
        sections.append(f"Could not perform power-law analysis: {str(e)}")
    
    # Additional Metrics
    sections.append("\n## 6. Additional Network Metrics\n")
    
    # Clustering coefficient
    try:
        if is_directed:
            avg_clustering = nx.average_clustering(G.to_undirected())
        else:
            avg_clustering = nx.average_clustering(G)
        sections.append(f"Average Clustering Coefficient: {avg_clustering:.4f}")
        
        # Transitivity
        transitivity = nx.transitivity(G)
        sections.append(f"Transitivity: {transitivity:.4f}")
    except Exception as e:
        sections.append(f"Could not compute clustering metrics: {str(e)}")
    
    # Assortativity
    try:
        if is_directed:
            # Out-Out degree assortativity
            assortativity = nx.degree_assortativity_coefficient(G, x='out', y='out')
            sections.append(f"Degree Assortativity Coefficient (out-out): {assortativity:.4f}")
        else:
            assortativity = nx.degree_assortativity_coefficient(G)
            sections.append(f"Degree Assortativity Coefficient: {assortativity:.4f}")
            
            if assortativity > 0.2:
                sections.append("This network shows assortative mixing (similar degree nodes tend to connect).")
            elif assortativity < -0.2:
                sections.append("This network shows disassortative mixing (different degree nodes tend to connect).")
            else:
                sections.append("This network shows relatively neutral degree mixing patterns.")
    except Exception as e:
        sections.append(f"Could not compute assortativity: {str(e)}")
    
    # Special graph properties
    sections.append("\n## 7. Special Graph Properties\n")
    
    properties_info = []
    
    # Check if it's a tree
    if not is_directed and nx.is_connected(G):
        is_tree = nx.is_tree(G)
        properties_info.append(f"Is a Tree: {is_tree}")
    
    # Check if it's bipartite
    try:
        is_bipartite = nx.is_bipartite(G)
        properties_info.append(f"Is Bipartite: {is_bipartite}")
    except:
        properties_info.append("Could not determine if graph is bipartite")
    
    # Check for self-loops
    has_selfloops = nx.number_of_selfloops(G) > 0
    properties_info.append(f"Has Self-loops: {has_selfloops}")
    
    # Check for multi-edges (if applicable)
    if hasattr(G, "is_multigraph") and G.is_multigraph():
        properties_info.append("Contains Multi-edges: True")
    
    # Check if graph is a complete graph (if not too large)
    if G.number_of_nodes() <= 1000:
        if not is_directed:
            is_complete = nx.is_connected(G) and nx.density(G) == 1.0
            properties_info.append(f"Is Complete Graph: {is_complete}")
        else:
            # A complete directed graph has n(n-1) edges
            n = G.number_of_nodes()
            is_complete = G.number_of_edges() == n * (n - 1)
            properties_info.append(f"Is Complete Directed Graph: {is_complete}")
    
    sections.append("\n".join(properties_info) + "\n")
    
    # Weight Analysis (if weighted)
    if is_weighted:
        sections.append("\n## 8. Edge Weight Analysis\n")
        
        # Get all edge weights
        weights = [w for _, _, w in G.edges(data='weight')]
        
        # Weight statistics
        weight_stats = [
            f"Edge Weight Statistics:",
            f"  - Mean weight: {np.mean(weights):.4f}",
            f"  - Median weight: {np.median(weights):.4f}",
            f"  - Max weight: {max(weights):.4f}",
            f"  - Min weight: {min(weights):.4f}",
            f"  - Standard deviation: {np.std(weights):.4f}"
        ]
        sections.append("\n".join(weight_stats) + "\n")
        
        # Weight distribution
        weight_distribution = Counter([round(w, 2) for w in weights])
        sections.append("\nWeight distribution (top 10 most common weights):")
        
        for weight, count in weight_distribution.most_common(10):
            sections.append(f"  - Weight {weight}: {count} edges ({count/len(weights)*100:.2f}%)")
    
    # Generate visualizations if requested
    if include_visualizations:
        sections.append("\n## 9. Visualizations\n")
        sections.append("The following visualizations have been generated:\n")
        
        # Create visualizations directory if it doesn't exist
        import os
        os.makedirs(vis_output_dir, exist_ok=True)
        
        # 1. Basic graph visualization
        plt.figure(figsize=(12, 8))
        if G.number_of_nodes() <= 1000:
            # For smaller graphs, use spring layout
            pos = nx.spring_layout(G, seed=42)
            nx.draw_networkx(G, pos, 
                            node_size=50,
                            node_color='lightblue',
                            edge_color='gray',
                            with_labels=G.number_of_nodes() <= 50,
                            arrows=is_directed)
        else:
            # For larger graphs, just show a sample
            sampled_nodes = np.random.choice(list(G.nodes()), size=min(1000, G.number_of_nodes()), replace=False)
            sampled_graph = G.subgraph(sampled_nodes)
            pos = nx.spring_layout(sampled_graph, seed=42)
            nx.draw_networkx(sampled_graph, pos, 
                            node_size=50,
                            node_color='lightblue',
                            edge_color='gray',
                            with_labels=False,
                            arrows=is_directed)
            plt.title(f"Network Visualization (Sample of {len(sampled_nodes)} nodes)")
        
        plt.title("Network Visualization")
        plt.axis('off')
        vis_path = os.path.join(vis_output_dir, "graph_visualization.png")
        plt.savefig(vis_path)
        plt.close()
        sections.append(f"  - Basic network visualization: {vis_path}")
        
        # 2. Degree distribution
        plt.figure(figsize=(10, 6))
        if is_directed:
            in_degrees = [d for _, d in G.in_degree()]
            plt.hist(in_degrees, bins=30, alpha=0.7, label='In-degree')
            out_degrees = [d for _, d in G.out_degree()]
            plt.hist(out_degrees, bins=30, alpha=0.7, label='Out-degree')
            plt.legend()
        else:
            degrees = [d for _, d in G.degree()]
            plt.hist(degrees, bins=30)
        
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution')
        plt.grid(True, alpha=0.3)
        vis_path = os.path.join(vis_output_dir, "degree_distribution.png")
        plt.savefig(vis_path)
        plt.close()
        sections.append(f"  - Degree distribution histogram: {vis_path}")
        
        # 3. Log-log plot for power law analysis
        plt.figure(figsize=(10, 6))
        if is_directed:
            degree_sequence = sorted([d for _, d in G.in_degree()], reverse=True)
            plt.title('Log-Log Plot of In-Degree Distribution')
        else:
            degree_sequence = sorted([d for _, d in G.degree()], reverse=True)
            plt.title('Log-Log Plot of Degree Distribution')
            
        degree_count = Counter(degree_sequence)
        x = list(degree_count.keys())
        y = list(degree_count.values())
        
        # Filter out zeros for log plot
        x_filtered = [d for d in x if d > 0]
        y_filtered = [degree_count[d] for d in x_filtered]
        
        if x_filtered and y_filtered:
            plt.loglog(x_filtered, y_filtered, 'bo')
            plt.xlabel('Degree (log scale)')
            plt.ylabel('Count (log scale)')
            plt.grid(True, which="both", ls="-", alpha=0.3)
            vis_path = os.path.join(vis_output_dir, "degree_loglog_plot.png")
            plt.savefig(vis_path)
            plt.close()
            sections.append(f"  - Log-log plot of degree distribution: {vis_path}")
        
        # 4. Community visualization (if communities were detected)
        try:
            if 'partition' in locals():
                plt.figure(figsize=(12, 8))
                if G.number_of_nodes() <= 1000:
                    # Color nodes by community
                    cmap = plt.cm.get_cmap('tab20', max(partition.values()) + 1)
                    nx.draw_networkx(G, pos, 
                                    node_color=[partition[node] for node in G.nodes()],
                                    cmap=cmap,
                                    node_size=50,
                                    edge_color='gray',
                                    with_labels=G.number_of_nodes() <= 50,
                                    arrows=is_directed)
                else:
                    # For larger graphs, sample as before
                    sampled_nodes = np.random.choice(list(G.nodes()), size=min(1000, G.number_of_nodes()), replace=False)
                    sampled_graph = G.subgraph(sampled_nodes)
                    pos = nx.spring_layout(sampled_graph, seed=42)
                    
                    # Only use communities for sampled nodes
                    sampled_partition = {node: partition[node] for node in sampled_nodes if node in partition}
                    
                    cmap = plt.cm.get_cmap('tab20', max(partition.values()) + 1)
                    nx.draw_networkx(sampled_graph, pos, 
                                    node_color=[partition[node] for node in sampled_graph.nodes() if node in partition],
                                    cmap=cmap,
                                    node_size=50,
                                    edge_color='gray',
                                    with_labels=False,
                                    arrows=is_directed)
                
                plt.title("Community Structure Visualization")
                plt.axis('off')
                vis_path = os.path.join(vis_output_dir, "community_visualization.png")
                plt.savefig(vis_path)
                plt.close()
                sections.append(f"  - Community structure visualization: {vis_path}")
        except Exception as e:
            sections.append(f"  - Could not create community visualization: {str(e)}")
        
    # Compile the full report
    report = "\n".join(sections)
    
    # Save the report if requested
    if save_report:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_file}")
    
    return report

def is_weighted_graph(G):
    """Check if the graph is weighted (has 'weight' attribute on edges)"""
    if not G.edges:
        return False
    
    # Check a sample of edges (up to 100)
    edge_sample = list(G.edges(data=True))[:100]
    for _, _, data in edge_sample:
        if 'weight' in data:
            return True
    return False

def count_diamond_patterns(G):
    """
    Count the number of diamond patterns in the graph.
    
    A diamond pattern is a cycle of length 4 (a 4-cycle).
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to analyze
        
    Returns:
    --------
    int
        The number of diamond patterns
    """
    # For directed graphs, consider the undirected version for diamond patterns
    if G.is_directed():
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
    
    # Method 1: For smaller graphs, use cycle finding
    if G.number_of_nodes() <= 1000:
        diamond_count = 0
        # Find all cycles of length 4
        for cycle in nx.simple_cycles(nx.DiGraph(G_undirected)) if nx.is_directed(G_undirected) else nx.cycle_basis(G_undirected):
            if len(cycle) == 4:
                diamond_count += 1
        
        # Since cycle_basis finds a basis, we need to adjust for possible overcounting
        # or undercounting of diamonds. This is a simple approximation.
        return diamond_count
    
    # Method 2: For larger graphs, use a sampling approach or node-based counting
    else:
        # Sample a subset of nodes to check for diamonds
        sample_size = min(1000, G.number_of_nodes())
        sampled_nodes = np.random.choice(list(G.nodes()), size=sample_size, replace=False)
        sampled_graph = G_undirected.subgraph(sampled_nodes)
        
        diamond_count = 0
        for node in sampled_graph:
            neighbors = list(sampled_graph.neighbors(node))
            # For each pair of neighbors
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    n1, n2 = neighbors[i], neighbors[j]
                    # Check if these two neighbors share another neighbor besides the original node
                    common_neighbors = set(sampled_graph.neighbors(n1)) & set(sampled_graph.neighbors(n2))
                    common_neighbors.discard(node)  # Remove the original node
                    diamond_count += len(common_neighbors)
        
        # Each diamond is counted 4 times (once for each node in the diamond)
        diamond_count //= 4
        
        # Scale up to estimate the full graph
        scaling_factor = G.number_of_nodes() / sample_size
        estimated_diamond_count = int(diamond_count * (scaling_factor ** 2))
        
        return estimated_diamond_count

def get_node_sample(G, max_nodes=10):
    """Return a sample of nodes based on highest degree centrality"""
    if G.number_of_nodes() <= max_nodes:
        return list(G.nodes())
    
    # Get top nodes by degree centrality
    degree_centrality = nx.degree_centrality(G)
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in sorted_nodes[:max_nodes//2]]
    
    # Add some random nodes for diversity
    remaining_nodes = [node for node in G.nodes() if node not in top_nodes]
    if remaining_nodes:
        random_nodes = np.random.choice(remaining_nodes, 
                                        size=min(max_nodes - len(top_nodes), len(remaining_nodes)), 
                                        replace=False)
        return list(top_nodes) + list(random_nodes)
    else:
        return top_nodes

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import community as community_louvain  # python-louvain package
from tabulate import tabulate  # For pretty-printing tables

if __name__ == "__main__":
    #G = nx.read_edgelist("graph_edgelist", delimiter="|")
    #print("Diamond Count:", count_diamond_patterns(G))
    #analyze_graph(G)
    #exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_document", default=0, type=int, help= "Which document in the data directory to start at?")
    parser.add_argument("--end_document", default=30, type=int, help= "Which document in the data directory to end at?")
    args = parser.parse_args()
    start = time.time()
    graph_creator = GraphCreator(None)
    graph_creator.parse_data(args)
    graph_creator.create_kg()

    graph_creator.save_kg()

    print("Total Time:", time.time() - start)
    G = nx.read_edgelist("graph_edgelist", delimiter="|")
    print("Diamond Count:", count_diamond_patterns(G))
    analyze_graph(G)

 