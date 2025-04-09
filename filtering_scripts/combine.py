import glob

with open("combined_graph_edgelist.txt", "w") as outfile:
    for file in glob.glob("filtered_edgelists/filtered_edgelist_*"):
        with open(file, "r") as f:
            outfile.write(f.read())