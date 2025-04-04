import os
import networkx as nx
import json
from collections import defaultdict, deque
from tqdm import tqdm

def bfs_with_path_tracking(G, start, max_depth=None):
    """
    Performs BFS from a start node, tracking the paths and distances.
    Much more efficient than all_simple_paths for our purpose.
    
    Parameters:
    G (nx.DiGraph): The directed graph
    start: The starting node
    max_depth (int, optional): Maximum path length, if None no limit is applied
    
    Returns:
    dict: Maps each reachable node to a tuple of (distance, first_path)
    """
    visited = {start: (0, [start])}  # Maps node -> (distance, path)
    queue = deque([(start, 0, [start])])  # (node, distance, path)
    
    while queue:
        node, distance, path = queue.popleft()
        
        # Stop if we've reached the maximum depth
        if max_depth is not None and distance >= max_depth:
            continue
        
        for successor in G.successors(node):
            new_distance = distance + 1
            new_path = path + [successor]
            
            # If we haven't seen this node or found a shorter path
            if successor not in visited or new_distance < visited[successor][0]:
                visited[successor] = (new_distance, new_path)
                queue.append((successor, new_distance, new_path))
    
    # Remove the start node from the results
    if start in visited:
        del visited[start]
        
    return visited

def find_diamonds_efficient(G, source, max_depth=None):
    """
    Find diamond patterns starting from a specific source node.
    Much more efficient than the original approach.
    
    Parameters:
    G (nx.DiGraph): The directed graph
    source: The source node to start from
    max_depth (int, optional): Maximum path length, if None no limit
    
    Yields:
    dict: Diamond pattern details
    """
    # Find successors of the source node
    successors = list(G.successors(source))
    
    # Need at least 2 successors to form a diamond
    if len(successors) < 2:
        return
    
    # Compute distance and a sample path for each reachable node from each successor
    paths_from_successors = {}
    for successor in successors:
        paths_from_successors[successor] = bfs_with_path_tracking(G, successor, max_depth)
    
    # For each pair of successors, find common reachable nodes (terminal candidates)
    for i in range(len(successors)):
        for j in range(i+1, len(successors)):
            node1 = successors[i]
            node2 = successors[j]
            
            # Find nodes reachable from both successors
            reachable1 = set(paths_from_successors[node1].keys())
            reachable2 = set(paths_from_successors[node2].keys())
            terminal_candidates = reachable1.intersection(reachable2)
            
            # Process each terminal candidate
            for terminal in terminal_candidates:
                # Skip if the terminal is one of the immediate successors
                if terminal == node1 or terminal == node2:
                    continue
                
                # Get the first discovered path from each successor to the terminal
                dist1, path1 = paths_from_successors[node1][terminal]
                dist2, path2 = paths_from_successors[node2][terminal]
                
                # Create the complete paths including the source
                full_path1 = [source] + path1
                full_path2 = [source] + path2
                
                # Extract edge relations
                edges1 = []
                for k in range(len(full_path1) - 1):
                    u, v = full_path1[k], full_path1[k+1]
                    relation = G.edges[u, v].get('relation', '')
                    edges1.append({
                        'source': u,
                        'target': v,
                        'relation': relation
                    })
                
                edges2 = []
                for k in range(len(full_path2) - 1):
                    u, v = full_path2[k], full_path2[k+1]
                    relation = G.edges[u, v].get('relation', '')
                    edges2.append({
                        'source': u,
                        'target': v,
                        'relation': relation
                    })
                
                # Create diamond pattern
                diamond = {
                    'source': source,
                    'branch1': {
                        'nodes': full_path1,
                        'edges': edges1
                    },
                    'branch2': {
                        'nodes': full_path2,
                        'edges': edges2
                    },
                    'terminal': terminal
                }
                
                yield diamond

def find_all_diamonds(G, max_depth=None, yield_results=False):
    """
    Find all diamond patterns in a graph.
    
    Parameters:
    G (nx.DiGraph): The directed graph
    max_depth (int, optional): Maximum path length, if None no limit is applied
    yield_results (bool): If True, yield results one by one to save memory
    
    Returns or Yields:
    If yield_results is False, returns a list of all diamonds
    If yield_results is True, yields diamonds one by one
    """
    if yield_results:
        # Memory-efficient generator mode
        source_nodes = list(G.nodes())
        for source in tqdm(source_nodes, desc="Finding diamonds", unit="node"):
            for diamond in find_diamonds_efficient(G, source, max_depth):
                yield diamond
    else:
        # Collect all results
        all_diamonds = []
        source_nodes = list(G.nodes())
        for source in tqdm(source_nodes, desc="Finding diamonds", unit="node"):
            diamonds = list(find_diamonds_efficient(G, source, max_depth))
            all_diamonds.extend(diamonds)
        return all_diamonds

def process_edgelist_files(folder_path, output_folder=None, max_depth=None, 
                      start_index=None, end_index=None, memory_efficient=False):
    """
    Process all edgelist files in the specified folder,
    find diamond patterns, and save the results to individual files.
    
    Parameters:
    folder_path (str): Path to the folder containing edgelist files
    output_folder (str, optional): Path to folder where result files will be saved
    max_depth (int, optional): Maximum path length to consider
    start_index (int, optional): Starting index of files to process
    end_index (int, optional): Ending index of files to process
    memory_efficient (bool): If True, process and save diamonds one by one
    
    Returns:
    dict: A dictionary containing summary information for all processed files
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    # Create output folder if it doesn't exist
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    summary = {}
    
    # Get sorted list of files
    file_list = sorted([f for f in os.listdir(folder_path) 
                      if os.path.isfile(os.path.join(folder_path, f))])
    
    # Apply index slicing if specified
    if start_index is not None or end_index is not None:
        start = start_index if start_index is not None else 0
        end = end_index if end_index is not None else len(file_list)
        file_list = file_list[start:end]
        print(f"Processing files from index {start} to {end-1}")
    
    # Process each file in the folder with a progress bar
    for filename in tqdm(file_list, desc="Processing Files", unit="file"):
        filepath = os.path.join(folder_path, filename)
            
        try:
            # Load the edgelist with pipe delimiter
            G = nx.read_edgelist(filepath, delimiter="|", create_using=nx.DiGraph())
            
            print(f"Finding diamonds in {filename} ({len(G.nodes())} nodes, {len(G.edges())} edges)")
            
            # Memory-efficient processing
            if memory_efficient and output_folder:
                # Create output filename
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_diamonds.json"
                output_path = os.path.join(output_folder, output_filename)
                
                # Process and write diamonds one by one
                with open(output_path, 'w') as f:
                    # Start the JSON file
                    f.write('{\n')
                    f.write(f'  "graph_info": {{\n')
                    f.write(f'    "nodes": {len(G.nodes())},\n')
                    f.write(f'    "edges": {len(G.edges())}\n')
                    f.write('  },\n')
                    f.write('  "diamonds": [\n')
                    
                    # Stream diamonds one by one
                    diamond_count = 0
                    for i, diamond in enumerate(find_all_diamonds(G, max_depth=max_depth, yield_results=True)):
                        if i > 0:
                            f.write(',\n')
                        json.dump(diamond, f, indent=4)
                        diamond_count += 1
                        if diamond_count % 1000 == 0:
                            print("Diamond Count", diamond_count, flush=True)
                        if diamond_count == 10000:
                            exit(0)
                    
                    # Close the JSON structure
                    f.write('\n  ],\n')
                    f.write(f'  "diamond_count": {diamond_count}\n')
                    f.write('}\n')
                
                print(f"Results for {filename} saved to {output_path}")
                
                # Store summary information
                summary[filename] = {
                    'nodes': len(G.nodes()),
                    'edges': len(G.edges()),
                    'diamond_count': diamond_count
                }
                
                print(f"Processed {filename}: Found {diamond_count} diamond patterns")
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                
            else:
                # Standard processing (load all diamonds into memory)
                diamonds = find_all_diamonds(G, max_depth=max_depth)
                
                # Create results for this file
                file_results = {
                    'graph_info': {
                        'nodes': len(G.nodes()),
                        'edges': len(G.edges())
                    },
                    'diamonds': diamonds,
                    'diamond_count': len(diamonds)
                }
                
                # Save results to individual file if output folder is specified
                if output_folder:
                    base_name = os.path.splitext(filename)[0]
                    output_filename = f"{base_name}_diamonds.json"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    with open(output_path, 'w') as f:
                        json.dump(file_results, f, indent=2)
                    print(f"Results for {filename} saved to {output_path}")
                
                # Store summary information
                summary[filename] = {
                    'nodes': len(G.nodes()),
                    'edges': len(G.edges()),
                    'diamond_count': len(diamonds)
                }
                
                print(f"Processed {filename}: Found {len(diamonds)} diamond patterns")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            summary[filename] = {'error': str(e)}
    
    # Save summary to the output folder if specified
    if output_folder:
        summary_path = os.path.join(output_folder, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_path}")
    
    return summary

def print_diamond_summary(summary):
    """
    Print a summary of diamond patterns found in each file.
    """
    total_diamonds = 0
    
    print("\nSummary of Diamond Patterns:")
    print("=" * 50)
    
    for filename, data in summary.items():
        diamond_count = data.get('diamond_count', 0)
        total_diamonds += diamond_count
        
        if 'error' in data:
            print(f"{filename}: Error - {data['error']}")
        else:
            print(f"{filename}: {diamond_count} diamonds (Nodes: {data['nodes']}, Edges: {data['edges']})")
    
    print("=" * 50)
    print(f"Total diamond patterns found: {total_diamonds}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Find diamond patterns in directed graphs from edgelist files.')
    parser.add_argument('--folder_path', type=str, help='Path to the folder containing edgelist files')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='Path to folder where result files will be saved (optional)')
    parser.add_argument('--max-depth', '-d', type=int, default=None, 
                        help='Maximum path length to consider for diamond patterns (optional)')
    parser.add_argument('--start-index', '-s', type=int, default=None,
                        help='Starting index of files to process (0-based)')
    parser.add_argument('--end-index', '-e', type=int, default=None,
                        help='Ending index of files to process (exclusive)')
    parser.add_argument('--memory-efficient', '-m', action='store_true',
                        help='Use memory-efficient processing (stream results directly to file)')
    parser.add_argument('--disable-progress', action='store_true', 
                        help='Disable progress bars (useful for logging to file)')
    
    args = parser.parse_args()
    
    folder_path = args.folder_path
    output_folder = args.output
    max_depth = args.max_depth
    start_index = args.start_index
    end_index = args.end_index
    memory_efficient = args.memory_efficient
    
    # Configure tqdm based on progress bar preference
    if args.disable_progress:
        # Monkey patch tqdm to do nothing
        global tqdm
        tqdm = lambda *args, **kwargs: args[0]
    
    print(f"Processing files in {folder_path}")
    print(f"Output folder: {output_folder if output_folder else 'None (results will not be saved)'}")
    print(f"Maximum path depth: {max_depth if max_depth else 'No limit'}")
    print(f"File index range: {start_index if start_index is not None else 0} to " +
          f"{end_index if end_index is not None else 'end'}")
    print(f"Memory-efficient mode: {'Enabled' if memory_efficient else 'Disabled'}")
    
    # Process the files and find diamonds
    summary = process_edgelist_files(
        folder_path, 
        output_folder, 
        max_depth=max_depth,
        start_index=start_index,
        end_index=end_index,
        memory_efficient=memory_efficient
    )
    
    # Print summary
    print_diamond_summary(summary)

if __name__ == "__main__":
    main()