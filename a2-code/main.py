import heapq
import json
import math
import os
import random
import networkx as nx
import numpy as np

# Check if all files exist currently
if os.path.isfile('2005_collaboration_network.gexf') and os.path.isfile(
        '2006_collaboration_network.gexf') and os.path.isfile('2005_collaboration_network_weighted.gexf'):

    # Load the graph from the .gexf file
    G_2005 = nx.read_gexf('2005_collaboration_network.gexf')
    largest_cc_2005 = max(nx.connected_components(G_2005), key=len)
    GCC_2005 = G_2005.subgraph(largest_cc_2005).copy()

    # Load the graph from the .gexf file
    G_2005_weighted = nx.read_gexf('2005_collaboration_network_weighted.gexf')
    largest_cc_2005_weighted = max(nx.connected_components(G_2005_weighted), key=len)
    GCC_2005_weighted = G_2005_weighted.subgraph(largest_cc_2005_weighted).copy()

    # Load the graph from the .gexf file
    G_2006 = nx.read_gexf('2006_collaboration_network.gexf')
    largest_cc_2006 = max(nx.connected_components(G_2006), key=len)
    GCC_2006 = G_2005.subgraph(largest_cc_2006).copy()

    # Prints number of nodes and edges for each graph
    print("The number of nodes in the GCC of 2005 is:", GCC_2005.number_of_nodes())
    print("The number of edges in the GCC of 2005 is:", GCC_2005.number_of_edges())
    print("The number of nodes in the GCC of 2005_weighted is:", GCC_2005_weighted.number_of_nodes())
    print("The number of edges in the GCC of 2005_weighted is:", GCC_2005_weighted.number_of_edges())
    print("The number of nodes in the GCC of 2006 is:", GCC_2006.number_of_nodes())
    print("The number of edges in the GCC of 2006 is:", GCC_2006.number_of_edges())

# If any of the files are unavailable, find all the nodes and edges of all graphs
else:
    # Load the data from the json file
    with open('coauthorship.json', 'r') as f:
        data = json.loads(f.read())

    # Finds the collaboration networks for each year
    collaboration_network_2005 = {}
    collaboration_network_2006 = {}

    # For each line check if the year is correct, if so format the collaboration network accordingly
    for obj in data:
        year = obj[2]

        # If year is 2005, then add to the collaboration_network_2005
        if year == 2005:
            author1 = obj[0]
            author2 = obj[1]
            if author1 not in collaboration_network_2005:
                collaboration_network_2005[author1] = []
            if author2 not in collaboration_network_2005:
                collaboration_network_2005[author2] = []
            if author1 not in collaboration_network_2005[author2]:
                collaboration_network_2005[author2].append(author1)
            if author2 not in collaboration_network_2005[author1]:
                collaboration_network_2005[author1].append(author2)

        # If year is 2006, then add to the collaboration_network_2006
        elif year == 2006:
            author1 = obj[0]
            author2 = obj[1]
            if author1 not in collaboration_network_2006:
                collaboration_network_2006[author1] = []
            if author2 not in collaboration_network_2006:
                collaboration_network_2006[author2] = []
            if author1 not in collaboration_network_2006[author2]:
                collaboration_network_2006[author2].append(author1)
            if author2 not in collaboration_network_2006[author1]:
                collaboration_network_2006[author1].append(author2)

    # Create empty graphs
    G_2005 = nx.Graph()
    G_2005_weighted = nx.Graph()
    G_2006 = nx.Graph()

    # Add nodes to the graph from the collaboration_network_2005
    for author in collaboration_network_2005:
        G_2005.add_node(author)
        G_2005_weighted.add_node(author)

    # Add nodes to the graph from the collaboration_network_2006
    for author in collaboration_network_2006:
        G_2006.add_node(author)

    # A dictionary used to keep the weights of the weighted 2005 graph
    edge_weights = {}

    # Add edges to the 2005 graph
    for author1 in collaboration_network_2005:
        for author2 in collaboration_network_2005[author1]:
            G_2005.add_edge(author1, author2)

            # Adds weights to the weighted graph dictionary
            if author1 != author2:
                if (author1, author2) in edge_weights:
                    edge_weights[(author1, author2)] += 1
                else:
                    edge_weights[(author1, author2)] = 1

    # Adds edges to the weighted 2005 graph
    for edge, weight in edge_weights.items():
        G_2005_weighted.add_edge(author1, author2, weight=weight)

    # Add edges to the 2006 graph
    for author1 in collaboration_network_2006:
        for author2 in collaboration_network_2006[author1]:
            G_2006.add_edge(author1, author2)

    # Finds largest connected component
    largest_cc_2005 = max(nx.connected_components(G_2005), key=len)
    largest_cc_2005_weighted = max(nx.connected_components(G_2005_weighted), key=len)
    largest_cc_2006 = max(nx.connected_components(G_2006), key=len)

    # Make a copy of the graph that only has the greatest connected component.
    GCC_2005 = G_2005.subgraph(largest_cc_2005).copy()
    GCC_2005_weighted = G_2005.subgraph(largest_cc_2005).copy()
    GCC_2006 = G_2006.subgraph(largest_cc_2006).copy()

    # Save the graph as a .gexf file
    nx.write_gexf(GCC_2005, '2005_collaboration_network.gexf')
    nx.write_gexf(GCC_2005_weighted, '2005_collaboration_network_weighted.gexf')
    nx.write_gexf(GCC_2006, '2006_collaboration_network.gexf')

# Calculate PageRank for 2005
pr = nx.pagerank(GCC_2005, alpha=0.85, personalization=None, max_iter=100, nstart=None, weight='weight')

# Round the PageRank scores to 3 significant figures
rounded_pr = np.round(list(pr.values()), 7)

# Finds top 50 pr_scores
pr_scores = sorted([(author, pr_score) for author, pr_score in zip(sorted(GCC_2005.nodes()), rounded_pr)],
                   key=lambda x: x[1], reverse=True)[:50]

# Open a text file and write the PageRank scores
with open('2005_pagerank_scores.txt', 'w') as f:
    for author, pr_score in pr_scores:
        f.write(f"{author}: {np.format_float_scientific(pr_score, precision=7, trim='k')}\n")

print("PageRank scores written to 2005_pagerank_scores.txt")

# Calculate the edge betweenness scores
edge_betweenness_scores = nx.edge_betweenness_centrality(GCC_2005, k=10)

# Round the edge betweenness scores to four significant figures
rounded_edge_betweenness_scores = {edge: round(score, 8) for edge, score in edge_betweenness_scores.items()}

# Sort the edges by their rounded edge betweenness scores in descending order
sorted_edges = sorted(rounded_edge_betweenness_scores.items(), key=lambda x: x[1], reverse=True)

# Save the top 20 edges to a text file
with open("2005_edge_betweenness_scores.txt", "w") as f:
    for edge, score in sorted_edges[:20]:
        f.write(f"{edge[0]} - {edge[1]}: {score}\n")

print("Edge betweenness scores and authors written to 2005_edge_betweenness_scores.txt")

# Calculate the degree of each node in the graph
degrees_2005 = dict(GCC_2005.degree())
degrees_2006 = dict(GCC_2006.degree())

# Create a new graph that contains only the nodes with more than 3 degrees
dblp2005_core = nx.Graph()
dblp2006_core = nx.Graph()

# Finds the nodes in the 2005 graph that has 3 or more degrees.
# Save the found nodes and edges to the 2005_core.
for node in degrees_2005:
    if degrees_2005[node] >= 3:
        dblp2005_core.add_node(node)
        for neighbor in GCC_2005.neighbors(node):
            if degrees_2005[neighbor] >= 3:
                dblp2005_core.add_edge(node, neighbor)

# Finds the nodes in the 2005 graph that has 3 or more degrees.
# Save the found nodes and edges to the 2006_core.
for node in degrees_2006:
    if degrees_2006[node] >= 3:
        dblp2006_core.add_node(node)
        for neighbor in GCC_2006.neighbors(node):
            if degrees_2006[neighbor] >= 3:
                dblp2006_core.add_edge(node, neighbor)

    # Prints number of nodes and edges for each graph
print("The number of nodes in the dblp2005_core is:", dblp2005_core.number_of_nodes())
print("The number of edges in the dblp2005_core is:", dblp2005_core.number_of_edges())
print("The number of nodes in the dblp2006_core is:", dblp2006_core.number_of_nodes())
print("The number of edges in the dblp2006_core is:", dblp2006_core.number_of_edges())

# Load the friends-of-friends set from a file if it exists
# fof_file = "fof.txt"
# if os.path.exists(fof_file):
#     with open(fof_file, "r") as f:
#         fof = {(line.strip().split("-")[0], line.strip().split("-")[1]) for line in f.readlines()}
# else:

# Initialize the set of friends-of-friends (FoF)
fof = set()

# Compute the set of friends-of-friends (FoF)
for node1 in dblp2005_core.nodes():

    # Get the neighbors of node1
    neighbors_node1 = [node for node in dblp2005_core.neighbors(node1)]

    for neighbor in neighbors_node1:
        # Get the neighbors of neighbor
        neighbors_neighbor = [node for node in dblp2005_core.neighbors(neighbor)]

        # Check if there are any common neighbors between node1 and neighbor
        for common_neighbor in set(neighbors_node1) & set(neighbors_neighbor):
            if common_neighbor != node1 and (node1, common_neighbor) not in fof:
                fof.add((node1, common_neighbor))

# # Save the friends-of-friends set to a file
# with open(fof_file, "w") as f:
#     for edge in sorted(list(set(fof))):  # Convert the set to a list and remove duplicates
#         f.write(f"{edge[0]}-{edge[1]}\n")

# Compute the set of edges that do not exist in dblp2005_core but exist in dblp2006_core
dblp2005_core_edges = set(dblp2005_core.edges)
dblp2006_core_edges = set(dblp2006_core.edges)

# Compute the set of edges that do not exist in dblp2005_core but exist in dblp2006_core
# There is some strange random deviation in this number
target_edges = dblp2006_core_edges - dblp2005_core_edges
print("The number of target edges is: ", len(target_edges))

# Define the values of k
k_values = [10, 20, 50, 100, len(target_edges)]

# Change the graph into a DiGraph to find the out_degrees
G = nx.DiGraph(dblp2005_core)

# Out degrees of each node
out_degrees = dict(G.out_degree)

# Find the maximum out-degree
max_out_degree = max(out_degrees.values())

# Dictionaries of scores
fof_common_neighbor_scores = {}
jaccard_scores = {}
pas_scores = {}
adamic_scores = {}

# This loop is different than using the methods. It is much faster and specifically wrote for this application.
# We need to loop over the names in friends-of-friends, but we need data from dblp2005_core.
# Other methods struggled to do this, so I wrote it myself.
for node1, node2 in fof:
    # Common Neighbor Scores
    common_neighbor_intersection = set(dblp2005_core.neighbors(node1)) & set(dblp2005_core.neighbors(node2))
    fof_common_neighbor_score = len(common_neighbor_intersection)
    fof_common_neighbor_scores[(node1, node2)] = fof_common_neighbor_score

    # Jaccard Neighbor Scores
    common_neighbor_union = set(dblp2005_core.neighbors(node1)) | set(dblp2005_core.neighbors(node2))
    fof_common_neighbor_union = len(common_neighbor_union)
    jaccard_coefficient = fof_common_neighbor_score / fof_common_neighbor_union
    jaccard_scores[(node1, node2)] = jaccard_coefficient

    # Preferential Attachment Scores
    pas_score = (math.log(out_degrees[node1]) + math.log(out_degrees[node2])) / math.log(max_out_degree)
    pas_scores[(node1, node2)] = pas_score

    # Adamic-Adar Scores
    adamic_score = 0
    for neighbor in common_neighbor_intersection:
        if out_degrees[neighbor] == 0:
            adamic_score += 0
        else:
            adamic_score += 1 / math.log(out_degrees[neighbor])
    adamic_scores[(node1, node2)] = adamic_score

# Run the prediction code for each value of k
for k in k_values:
    # Select a random subset of FoF to predict as edges
    predicted_edges = random.sample(list(fof), k)

    # Convert the predicted_edges to a set of edges
    P = set([edge for edge in predicted_edges])

    # Calculate the precision at k
    precision_RD = len(P.intersection(target_edges)) / len(P)
    rounded_precision_RD = format(precision_RD, ".6f")

    # Finds the best nodes
    best_nodes_CN = heapq.nlargest(k, fof_common_neighbor_scores.keys(), key=lambda x: fof_common_neighbor_scores[x])

    # Convert the predicted common neighbors to a set
    P_CN = set(best_nodes_CN)

    # Calculate the precision for the common neighbors
    precision_CN = len(P_CN.intersection(target_edges)) / len(P_CN)
    rounded_precision_CN = format(precision_CN, ".6f")

    # Finds the best nodes
    best_nodes_JC = heapq.nlargest(k, jaccard_scores.keys(), key=lambda x: jaccard_scores[x])

    # Convert the predicted jaccard nodes to a set
    P_JC = set(best_nodes_JC)

    # Calculate the precision for the common neighbors
    precision_JC = len(P_JC.intersection(target_edges)) / len(P_JC)
    rounded_precision_JC = format(precision_JC, ".6f")

    # Finds the best nodes
    best_nodes_PA = heapq.nlargest(k, pas_scores.keys(), key=lambda x: pas_scores[x])

    # Convert the predicted preferential attachment nodes to a set
    P_PA = set(best_nodes_PA)

    # Calculate the precision for the common neighbors
    precision_PA = len(P_PA.intersection(target_edges)) / len(P_PA)
    rounded_precision_PA = format(precision_PA, ".6f")

    # Finds the best nodes
    best_nodes_AA = heapq.nlargest(k, adamic_scores.keys(), key=lambda x: adamic_scores[x])

    # Convert the predicted adamic adar nodes to a set
    P_AA = set(best_nodes_AA)

    # Calculate the precision for the common neighbors
    precision_AA = len(P_AA.intersection(target_edges)) / len(P_AA)
    rounded_precision_AA = format(precision_AA, ".6f")

    # Varied format output
    if k == len(target_edges):
        print(f"RD for P@T is {rounded_precision_RD}")
        print(f"CN for P@T is {rounded_precision_CN}")
        print(f"JC for P@T is {rounded_precision_JC}")
        print(f"PA for P@T is {rounded_precision_PA}")
        print(f"AA for P@T is {rounded_precision_AA}")
    # Typical format output
    else:
        print(f"RD for P@{k} is {rounded_precision_RD}")
        print(f"CN for P@{k} is {rounded_precision_CN}")
        print(f"JC for P@{k} is {rounded_precision_JC}")
        print(f"PA for P@{k} is {rounded_precision_PA}")
        print(f"AA for P@{k} is {rounded_precision_AA}")

# Using the louvain communities method to find communities.
# Girvan-Newman method was not working with such a large network.
partitions = list(nx.community.louvain_communities(GCC_2005))

if len(partitions) == 0:
    print("The Louvain partitions algorithm didn't return any communities.")
else:
    # Get the community sizes
    community_sizes = [len(c) for c in partitions]
    # Sort the community sizes in descending order
    community_sizes.sort(reverse=True)
    # Print the sizes of all communities
    print("Sizes of Top 10 communities:")
    for i in range(min(10, len(community_sizes))):
        print(f"Community {i + 1}: {community_sizes[i]} nodes")
