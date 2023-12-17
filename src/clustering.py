class PerformClustering:
    def __init__(self, max_number_of_clusters=50):
        self.max_number_of_clusters = max_number_of_clusters
        pass

    def plot_elbow_curve(self, dataset):
        print(self.max_number_of_clusters)
        return dataset

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
