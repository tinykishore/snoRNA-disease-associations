import pandas as pd
import numpy
from src.read_data import DataSet
from src.clustering import PerformClustering

# Section 1: Instantiate DataSet class
"""
Instantiating the DataSet class will create an object. This object will contain all the datasets:
    
    dataset_object = DataSet()
    
    - dataset_object.disease_name()                     [Checked]
    - dataset_object.disease_size()                     [Checked]
    - dataset_object.snoRNA_name()                      [Checked]
    - dataset_object.snoRNA_size()                      [Checked]
    - dataset_object.known_association()                [Checked]
    - dataset_object.SnoRNA_similarity()                [Checked]
    - dataset_object.disease_similarity()               [Checked]
    - dataset_object.disease_semantic_similarity()      [Has Issues]
    - dataset_object.adjacency_matrix()                 [Apparently Same, but ig has some issues]
    - dataset_object.snoRNA_functional_similarity()     [Has Issues]
    
    *** Issues: Size of dataset_object.disease_semantic_similarity is 59x59 instead of 60x60
                Size of dataset_object.snoRNA_functional_similarity is 570x570 instead of 571x571
                Size of dataset_object.adjacency_matrix is 570x59 instead of 571x60

"""

DataSet = DataSet()
# Print all the datasets to check

print(DataSet.disease_name)


# Section 2: Plot elbow curve to find the optimal number of clusters
"""
This function will plot the elbow curve for the given dataset and the given maximum number of clusters.
Then we can use the elbow curve to find the optimal number of clusters

    PerformClustering = PerformClustering(max_number_of_clusters= 50)
    
    - PerformClustering.plot_elbow_curve(major)
    - PerformClustering.prepare_data_for_clustering(
            adjacency_matrix, 
            disease_semantic_similarity, 
            snoRNA_functional_similarity
      )
    - 
"""
PerformClustering = PerformClustering(max_number_of_clusters=50)
# PerformClustering.plot_elbow_curve()
unknown, known, major = PerformClustering.prepare_data_for_clustering(
    DataSet.adjacency_matrix,
    DataSet.disease_semantic_similarity,
    DataSet.snoRNA_functional_similarity,
    DataSet.snoRNA_size,
    DataSet.disease_size
)
# major 33543 in base file but 32913 in this file [Has Issues]
# unknown 33543 in base file but 32913 in this file [Has Issues]
# known 33543 in base file but 717 in this file [Has Issues]


print(len(known))


# Section 3: Perform K-Means Clustering
"""
    This function will perform K-Means Clustering on the given dataset and the given number of clusters.
    Then it will return the clustered dataset.
"""
