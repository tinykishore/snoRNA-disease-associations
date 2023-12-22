from sklearn.cluster import KMeans


class PerformClustering:
    def __init__(self):
        self.number_of_clusters = 20

    def prepare_data_for_clustering(
            self,
            adjacency_matrix,
            disease_semantic_similarity,
            snoRNA_functional_similarity,
            snoRNA_size,
            disease_size):

        unknown = []
        known = []
        major = []

        for x in range(snoRNA_size):
            for y in range(disease_size):
                if adjacency_matrix[x, y] == 0:
                    unknown.append((x, y))
                else:
                    known.append((x, y))

        for z in range(len(unknown)):
            q = (disease_semantic_similarity[unknown[z][1], :].tolist()
                 + snoRNA_functional_similarity[unknown[z][0], :].tolist())
            major.append(q)

        return unknown, known, major

    def perform_clustering(self, dataset):

        # Initialize K-Means Clustering with the given number of clusters
        # also fit the dataset to the K-Means Clustering
        kmeans = (KMeans(
            n_clusters=self.number_of_clusters,
            random_state=0,
            n_init=10).fit(dataset))

        # Get the cluster centers
        center = kmeans.cluster_centers_

        center_x = []
        center_y = []
        for j in range(len(center)):
            center_x.append(center[j][0])
            center_y.append(center[j][1])

        # Get the labels for each data point
        labels = kmeans.labels_

        return labels
