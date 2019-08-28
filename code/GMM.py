__author__ = 'Manjunath Dharshan'
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

np.random.seed(10)
class GaussianMixtureModels:
    def __init__(self, n_components, covariances, mixing_probs):
        self.n_components = n_components
        self.means = None
        self.covariances = covariances
        self.mixing_probs = mixing_probs
        self.threshold = 1e-6
        self.__callback = None

    def fit(self, features):
        # Initialise
        n, _ = features.shape
        norm_densities = np.empty((n, self.n_components), np.float)
        responsibilities = np.empty((n, self.n_components), np.float)
        old_log_likelihood = 0
        self._initialise_parameters(features)

        while True:
            for i in np.arange(n):
                x = features[i]

                for j in np.arange(self.n_components):
                    norm_densities[i][j] = self.multivariate_normal_pdf(x, self.means[j], self.covariances[j])
            log_vector = np.log(np.array([np.dot(self.mixing_probs, norm_densities[i]) for i in np.arange(n)]))
            log_likelihood = np.dot(log_vector.T, np.ones(n))

            self.call_back()

            if np.absolute(log_likelihood - old_log_likelihood) < self.threshold:
                break

            for i in np.arange(n):
                x = features[i]
                denominator = np.dot(self.mixing_probs, norm_densities[i])
                for j in np.arange(self.n_components):
                    responsibilities[i][j] = self.mixing_probs[j] * norm_densities[i][j] / denominator

            for i in np.arange(self.n_components):
                responsibility = (responsibilities.T)[i]

                denominator = np.dot(responsibility.T, np.ones(n))

                # Update mean
                self.means[i] = np.dot(responsibility.T, features) / denominator

                self.mixing_probs[i] = denominator / n

            old_log_likelihood = log_likelihood

    def cluster(self, features):
        n, _ = features.shape
        partition = np.empty(n, np.int)
        distances = np.empty(self.n_components, np.float)
        cov_inverses = [np.linalg.inv(cov) for cov in self.covariances]

        for i in np.arange(n):
            x = features[i]
            for j in np.arange(self.n_components):
                distances[j] = np.dot(np.dot((x - self.means[j]).T, cov_inverses[j]), x - self.means[j])

            partition[i] = np.argmin(distances)

        return partition

    def call_back(self):
        if self.__callback:
            dct = {
                'mixing_probs': self.mixing_probs,
                'means': self.means,
                'covariances': self.covariances
            }
            self.__callback(dct)

    def multivariate_normal_pdf(self, x, mean, covariance):
        centered = x - mean
        cov_inverse = np.linalg.inv(covariance)
        cov_det = np.linalg.det(covariance)
        exponent = np.dot(np.dot(centered.T, cov_inverse), centered)
        return np.exp(-0.5 * exponent) / np.sqrt(cov_det * np.power(2 * np.pi, self.n_components))

    def _initialise_parameters(self, features):
        if not self.means or not self.covariances:
            n, m = features.shape

            indices = np.arange(n)
            np.random.shuffle(np.arange(n))
            features_shuffled = np.array([features[i] for i in indices])

            divs = int(np.floor(n / self.n_components))
            features_split = [features_shuffled[i:i + divs] for i in range(0, n, divs)]

            if not self.means:
                means = []
                for i in np.arange(self.n_components):
                    means.append(np.mean(features_split[i], axis=0))
                self.means = np.array(means)

            if not self.covariances:
                covariances = []
                for i in np.arange(self.n_components):
                    covariances.append(np.cov(features_split[i].T))
                self.covariances = np.array(covariances)

        if not self.mixing_probs:
            self.mixing_probs = np.repeat(1 / self.n_components, self.n_components)

    def plot_scatter(self,comparisons,dataset):
        print("comparisons::",comparisons)
        _1st_cluster_points = dataset[comparisons]
        _2nd_cluster_points = dataset[np.logical_not(comparisons)]
        figure = plt.figure(1)
        axes = figure.add_subplot(111)
        plt.title("GMM Clustering of Audio Data")
        plt.xlabel("1st Feature")
        plt.ylabel("2nd Feature")
        axes.scatter(_1st_cluster_points[:,0],_1st_cluster_points[:,1],c='#8B008B',marker='o',label='First Cluster')
        axes.scatter(_2nd_cluster_points[:, 0], _2nd_cluster_points[:, 1],c='#1E90FF',marker='*',label='Second Cluster')
        plt.legend(loc='best')
        plt.show()


if __name__ == '__main__':

   colors = ['r', 'b']

   dataset = np.genfromtxt('audioData.csv', delimiter=",")

   cov = np.cov(dataset.T)
   gmm = GaussianMixtureModels(n_components=2,covariances=[cov,cov],mixing_probs=[0.5,0.5])
   gmm.fit(dataset)
   p = gmm.cluster(dataset)
   p = ma.make_mask(p)
   gmm.plot_scatter(p,dataset)