# Link Analysis and Prediction; Community Detection

This project investigates the temporal dynamics and structural properties of co-authorship networks using the DBLP dataset for 2005 and 2006. Our analysis includes generating temporal snapshots, performing link analysis, predicting future collaborations, and detecting community structures.

## Objective and Motivation

The primary objective is to analyze the DBLP co-authorship network’s temporal dynamics and structural properties. This involves generating temporal snapshots, performing link analysis, predicting future collaborations, and detecting community structures to understand evolving co-authorship patterns over time.

## Methodology

### Temporal Graphs

	•	**dblp2005**: An undirected, unweighted graph for 2005.
	•	**dblp2006**: An undirected, unweighted graph for 2006.
	•	**dblp2005w**: A weighted version of dblp2005.
	•	**Greatest Connected Component (GCC)**: Extracted for each graph.

### Node and Edge Importance

	•	**PageRank Scores**: Calculated for dblp2005, dblp2006, and dblp2005w to identify key authors.
	•	**Edge Betweenness Scores**: Calculated to identify crucial connections.

### Link Prediction

	•	**Core Graph Construction**: Filtered nodes with degree ≥ 3.
	•	**Friends-of-Friends (FoF) Computation**: Identified potential new collaborations.
	•	**Target Edges Derivation**: Derived edges present in 2006 but absent in 2005.
	•	**Prediction Methods**: Implemented Random Predictor, Common Neighbors, Jaccard Coefficient, Preferential Attachment, and Adamic/Adar.
	•	**Precision at k**: Evaluated prediction methods.

### Community Detection

	•	**Louvain Method**: Applied to detect communities in dblp2005.

## Results

### Temporal Graphs

	•	**dblp2005**: 106,943 nodes, 300,043 edges.
	•	**dblp2006**: 50,879 nodes, 121,822 edges.
	•	**dblp2005w**: 106,943 nodes, 300,043 edges.

### Node and Edge Importance

	•	**PageRank Scores**: Identified top 50 authors.
	•	**Edge Betweenness Scores**: Identified top 20 edges.

### Link Prediction

	•	**Core Graphs**: dblp2005-core: 77,153 nodes, 255,815 edges. dblp2006-core: 32,948 nodes, 97,478 edges.
	•	**Precision Scores**:
	•	**Random Predictor (RD)**: Low precision.
	•	**Common Neighbors (CN)**: Precision score of 0.
	•	**Jaccard Coefficient (JC)**: Precision score of 0.
	•	**Preferential Attachment (PA)**: Positive for smaller k values.
	•	**Adamic/Adar (AA)**: Highest precision, particularly for smaller k values.

### Community Detection

	•	Identified top 10 community sizes using the Louvain method.

## Conclusion

This analysis of the DBLP co-authorship network provided valuable insights into its temporal dynamics and structural properties. By calculating PageRank and edge betweenness scores, we identified key authors and crucial connections. Evaluating link prediction methods revealed that Adamic/Adar and Preferential Attachment were the most effective. Community detection using the Louvain method highlighted the modular structure of the network. These findings aid in understanding co-authorship patterns, fostering research collaborations, and enhancing academic research growth.

## References

	•	NetworkX
	•	DBLP Dataset
	•	Louvain Method