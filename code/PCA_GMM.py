__author__ = 'Manjunath Dharshan'
import numpy as np
from numpy import linalg as la
import random
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

np.random.seed(10)
random.seed(10)

class PCA_GMM(object):
    def __init__(self):
        self.number_of_principal_components = 2
        self.number_of_clusters = 2
        self.prior_probabilities = self.number_of_clusters * [1 / 2]
        self.threshold = 1e-6

    def compute_principal_components(self):
        self.audio_data = np.genfromtxt('audioData.csv', delimiter=',')
        mean_of_audio_data = np.mean(self.audio_data,axis=0)
        centered_data = self.audio_data - mean_of_audio_data
        covariance_matrix = np.cov(centered_data, rowvar=False)
        eigen_values, eigen_vectors = la.eig(covariance_matrix)
        eigen_values_sorted=sorted(eigen_values,reverse=True)
        indices_of_top_eigen_vectors = np.where(eigen_values >= eigen_values_sorted[self.number_of_principal_components-1])
        top_eigen_vectors = eigen_vectors[indices_of_top_eigen_vectors]
        return top_eigen_vectors

    def project_data_along_principal_components(self,eigen_vectors):
        projected_data = np.dot(self.audio_data,eigen_vectors.T)
        self.audio_data = projected_data
        self.data_shape = self.audio_data.shape
        return projected_data

    def distance_from_centers(self,centroids,data):
        D2 = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in data])
        return D2

    def choose_next_center(self,D2,data):
        probabilities = D2 / D2.sum()
        cumulative_probabilities = probabilities.cumsum()
        r = random.random()
        ind = np.where(cumulative_probabilities >= r)[0][0]
        return data[np.array([ind])]

    def init_centers(self,k,data):
        random_centroid = np.random.choice(data.shape[0], 1, replace=False)

        centroids = data[random_centroid]
        while len(centroids) < k:
            D2 = self.distance_from_centers(centroids,data)
            an_array = self.choose_next_center(D2,data)
            centroids = np.append(centroids,an_array,axis=0)
        return centroids

    def initialize_parameters(self):
        self.centers = self.init_centers(2, self.audio_data)
        self.covariance_matrix = np.cov(self.audio_data, rowvar=False)

    def calculate_likelihoods(self):
        likelihoods = np.empty((2,128))
        for i in range(self.number_of_clusters):
            pdf = multivariate_normal.pdf(self.audio_data, mean=self.centers[i],cov=self.covariance_matrix)
            likelihoods[i] = pdf
        return likelihoods.T

    def em_algorithm(self):
        previous_log_likelihood = 0
        #while True:
        for _ in range(30):
            likelihoods = self.calculate_likelihoods()
            likelihoods_and_prior_probabilities = likelihoods * self.prior_probabilities
            point_probabilities = np.sum(likelihoods_and_prior_probabilities, axis=1)
            log_of_point_probabilities = np.log(point_probabilities)
            log_likelihood_of_data = np.sum(log_of_point_probabilities)
            normalized_scores = np.divide(likelihoods_and_prior_probabilities,
                                          point_probabilities[:, None])

            comparisons = normalized_scores[:, 0] >= normalized_scores[:, 1]
            if np.abs(log_likelihood_of_data - previous_log_likelihood) < self.threshold:
                break
            previous_log_likelihood=log_likelihood_of_data
            for i in range(len(self.centers)):
                sum_of_dimension_values = np.sum(normalized_scores[:, i])
                self.centers[i] = np.sum((normalized_scores[:,i][:,None] * self.audio_data), axis=0) / sum_of_dimension_values
                self.prior_probabilities[i] = sum_of_dimension_values / self.data_shape[0]
        return comparisons

    def plot_scatter(self,comparisons):
        _1st_cluster_points = self.audio_data[comparisons]
        _2nd_cluster_points = self.audio_data[np.logical_not(comparisons)]
        figure = plt.figure(1)
        axes = figure.add_subplot(111)
        plt.title("GMM Clustering of Audio Data")
        plt.xlabel("1st Feature")
        plt.ylabel("2nd Feature")
        axes.scatter(_1st_cluster_points[:,0],_1st_cluster_points[:,1],c='b',marker='.',label='First Cluster')
        axes.scatter(_2nd_cluster_points[:,0], _2nd_cluster_points[:,1],c='r',marker='.',label='Second Cluster')
        plt.legend(loc='lower left')
        plt.show()

if __name__ == '__main__':
    pca = PCA_GMM()
    eigen_vectors = pca.compute_principal_components()
    projected_data = pca.project_data_along_principal_components(eigen_vectors)
    pca.initialize_parameters()
    comparisons = pca.em_algorithm()
    pca.plot_scatter(comparisons)