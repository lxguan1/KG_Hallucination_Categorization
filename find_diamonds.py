import os
import networkx as nx
import json
import random
from collections import defaultdict, deque
from tqdm import tqdm

def bfs_with_path_tracking(G, start, max_depth=None):
    """
    Performs BFS from a start node, tracking the paths and distances.
    
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

def find_first_diamond(G, source, max_depth=None):
    """
    Find the first diamond pattern starting from a specific source node.
    
    Parameters:
    G (nx.DiGraph): The directed graph
    source: The source node to start from
    max_depth (int, optional): Maximum path length, if None no limit
    
    Returns:
    dict or None: Diamond pattern details or None if no diamond found
    """
    # Find successors of the source node
    successors = list(G.successors(source))
    
    # Need at least 2 successors to form a diamond
    if len(successors) < 2:
        return None
    
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
                
                # Check if paths share any nodes other than source and terminal
                path1_set = set(path1[:-1])
                path2_set = set(path2[:-1])
                if path1_set.intersection(path2_set):
                    # Paths share intermediate nodes, skip this diamond
                    continue

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
                
                return diamond
    
    return None

def find_unique_source_diamonds(G, max_depth=None, max_patterns=1000):
    """
    Find diamond patterns with unique source nodes.
    
    Parameters:
    G (nx.DiGraph): The directed graph
    max_depth (int, optional): Maximum path length, if None no limit is applied
    max_patterns (int): Maximum number of diamond patterns to find
    
    Returns:
    list: Diamond patterns with unique source nodes
    """
    diamonds = []
    
    # Get all potential source nodes (nodes with out-degree >= 2)
    potential_sources = [node for node in G.nodes() if G.out_degree(node) >= 2]
    
    # Shuffle the list to get a random order
    random.shuffle(potential_sources)
    
    # Limit the number of source nodes to check
    source_limit = min(max_patterns * 10, len(potential_sources))
    sources_to_check = potential_sources[:source_limit]
    
    print(f"Checking up to {len(sources_to_check)} potential source nodes for diamonds")
    
    # Process each source node
    for source in tqdm(sources_to_check, desc="Finding diamonds", unit="node"):
        # Stop if we've found enough patterns
        if len(diamonds) >= max_patterns:
            break
            
        # Find a diamond starting from this source
        diamond = find_first_diamond(G, source, max_depth)
        
        # If we found a diamond, add it to the list
        if diamond:
            diamonds.append(diamond)
    
    print(f"Found {len(diamonds)} diamond patterns with unique source nodes")
    return diamonds

def process_edgelist_files(folder_path, output_file, max_depth=None, max_patterns_per_graph=1000):
    """
    Process all edgelist files in the specified folder,
    find diamond patterns with unique source nodes, and save all results to one file.
    
    Parameters:
    folder_path (str): Path to the folder containing edgelist files
    output_file (str): Path to the output file
    max_depth (int, optional): Maximum path length to consider
    max_patterns_per_graph (int): Maximum number of diamond patterns to find per graph
    
    Returns:
    dict: A dictionary containing summary information for all processed files
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    summary = {}
    all_diamonds = []
    
    # Get sorted list of files
    file_list = sorted([f for f in os.listdir(folder_path) 
                      if os.path.isfile(os.path.join(folder_path, f))])
    
    # Process each file in the folder with a progress bar
    for filename in tqdm(file_list, desc="Processing Files", unit="file"):
        filepath = os.path.join(folder_path, filename)
            
        try:
            # Load the edgelist with pipe delimiter
            G = nx.read_edgelist(filepath, delimiter="|", create_using=nx.DiGraph())
            
            print(f"Finding diamonds in {filename} ({len(G.nodes())} nodes, {len(G.edges())} edges)")
            
            # Find diamonds with unique source nodes in this graph
            diamonds = find_unique_source_diamonds(
                G, 
                max_depth=max_depth, 
                max_patterns=max_patterns_per_graph
            )
            
            # Add file information to each diamond
            for diamond in diamonds:
                diamond['file'] = filename
            
            # Add to the overall collection
            all_diamonds.extend(diamonds)
            
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
    
    # Save all diamonds to a single file
    print(f"Saving {len(all_diamonds)} diamond patterns to {output_file}")
    with open(output_file, 'w') as f:
        # Create a combined results object
        combined_results = {
            'summary': summary,
            'total_diamonds': len(all_diamonds),
            'diamonds': all_diamonds
        }
        json.dump(combined_results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
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
    
    parser = argparse.ArgumentParser(description='Find diamond patterns with unique source nodes in directed graphs.')
    parser.add_argument('--folder_path', type=str, required=True,
                        help='Path to the folder containing edgelist files')
    parser.add_argument('--output', '-o', type=str, required=True, 
                        help='Path to the output file for all diamond patterns')
    parser.add_argument('--max-depth', '-d', type=int, default=None, 
                        help='Maximum path length to consider for diamond patterns (optional)')
    parser.add_argument('--patterns-per-graph', '-p', type=int, default=1000,
                        help='Maximum number of diamond patterns to find per graph (default: 1000)')
    parser.add_argument('--disable-progress', action='store_true', 
                        help='Disable progress bars (useful for logging to file)')
    
    args = parser.parse_args()
    
    folder_path = args.folder_path
    output_file = args.output
    max_depth = args.max_depth
    patterns_per_graph = args.patterns_per_graph
    
    # Configure tqdm based on progress bar preference
    if args.disable_progress:
        # Monkey patch tqdm to do nothing
        global tqdm
        tqdm = lambda *args, **kwargs: args[0]
    
    print(f"Processing files in {folder_path}")
    print(f"Output file: {output_file}")
    print(f"Maximum path depth: {max_depth if max_depth else 'No limit'}")
    print(f"Maximum patterns per graph: {patterns_per_graph}")
    
    # Process the files and find diamonds
    summary = process_edgelist_files(
        folder_path, 
        output_file, 
        max_depth=max_depth,
        max_patterns_per_graph=patterns_per_graph
    )
    
    # Print summary
    print_diamond_summary(summary)

if __name__ == "__main__":
    main()