import pandas as pd
import numpy


class DataSet:
    def __init__(self):
        self.disease_name = pd.read_csv('datasets/disease_name.csv', header=None)
        self.disease_size = len(self.disease_name)

        self.snoRNA_name = pd.read_csv('datasets/snoRNA_name.csv', header=None)
        self.snoRNA_size = len(self.snoRNA_name)

        self.known_association = pd.read_csv('datasets/known_association.csv', header=None)
        self.SnoRNA_similarity = pd.read_csv('datasets/snoRNA_similarity.csv', header=None)
        self.disease_similarity = pd.read_csv('datasets/disease_similarity.csv', header=None)

        self.disease_semantic_similarity = self.__prepare_disease_semantic_similarity()
        self.adjacency_matrix = self.__prepare_adjacency_matrix()
        self.snoRNA_functional_similarity = self.__prepare_snoRNA_functional_similarity()

    def __prepare_disease_semantic_similarity(self):
        disease_semantic_similarity = numpy.zeros(
            (self.disease_size, self.disease_size)
        )

        for i in range(len(self.disease_name)):
            for j in range(len(self.disease_name)):
                disease_semantic_similarity[i, j] = self.disease_similarity.iloc[i, j]
        return disease_semantic_similarity

    def __prepare_adjacency_matrix(self):
        adjacency_matrix = numpy.zeros(
            (self.snoRNA_size, self.disease_size)
        )

        for i in range(len(self.snoRNA_name)):
            for j in range(len(self.disease_name)):
                adjacency_matrix[i, j] = self.known_association.iloc[i, j]
        return adjacency_matrix

    def __prepare_snoRNA_functional_similarity(self):
        snoRNA_functional_similarity = numpy.zeros(
            (self.snoRNA_size, self.snoRNA_size)
        )

        for i in range(len(self.snoRNA_name)):
            for j in range(len(self.snoRNA_name)):
                snoRNA_functional_similarity[i, j] = self.SnoRNA_similarity.iloc[i, j]
        return snoRNA_functional_similarity
