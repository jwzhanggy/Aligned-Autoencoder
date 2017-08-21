from evaluate import *
from sklearn.metrics import *
from copy import *
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
import math

# this is my exp evaluate class, it inherits from class dataset
class expEvaluateInternal(evaluate):
    source_target = []
    type = ''
    dict = []
    social_links = []
    social_matrix = []
    sim_matrix = []
    K = 0
    
    def density(self, result, social_links, k):
        [m, n] = social_links.shape
        link_number = 0
        for i in range(m):
            for j in range(n):
                if social_links[i, j] != 0:
                    link_number = link_number + 1
        
        cluster_element_dict = {}
        for i in range(k):
            cluster_element_dict[i] = []
        
        for i in range(len(result)):
            cluster_index = result[i]
            cluster_element_dict[cluster_index].append(i)
            
        count_total = 0
        for i in range(k):
            element_list = cluster_element_dict[i]
            count_each_cluster = 0
            for m in element_list:
                for n in element_list:
                    if m != n and social_links[m, n] != 0.0:
                        count_each_cluster = count_each_cluster + 1
            count_total = count_total + count_each_cluster
        return float(count_total) / link_number
    
    def Silhouette_Coefficient(self, X, y, distance_type):
        return metrics.silhouette_score(X, y, metric='euclidean')
    
    def dbi(self, X, y, k):
        dbi_score = 0.0
        normalized_dbi_score = 0.0
        kmeans = KMeans(init='k-means++', n_clusters=k, max_iter=100, n_init=10, tol=1e-4)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        cluster_element_dict = {}
        for i in range(k):
            cluster_element_dict[i] = []
        
        for i in range(len(y)):
            cluster_index = y[i]
            cluster_element_dict[cluster_index].append(i)
        
        for i in range(k):
            dbi_list = []
            normalized_dbi_list = []
            distance_of_cluster_i = self.distance_within_cluster(X, cluster_element_dict[i])
            for j in range(k):
                if i == j:
                    continue
                distance_of_cluster_j = self.distance_within_cluster(X, cluster_element_dict[j])
                c_i = centers[i]
                c_j = centers[j]
                distance_of_centers = self.Euc_distance(c_i, c_j)
                if (distance_of_cluster_i + distance_of_cluster_j) != 0.0:
                    dbi_list.append(distance_of_centers / (distance_of_cluster_i + distance_of_cluster_j))
                else:
                    dbi_list.append(0.0)
                if (distance_of_cluster_i + distance_of_cluster_j + 2.0 * distance_of_centers) != 0.0:
                    normalized_dbi_list.append(2.0 * distance_of_centers / (2.0 * distance_of_centers + distance_of_cluster_i + distance_of_cluster_j))
                else:
                    normalized_dbi_list.append(0.0)
            dbi_list.append(0.0)
            normalized_dbi_list.append(0.0)
            
            dbi_score = dbi_score + max(dbi_list)
            normalized_dbi_score = normalized_dbi_score + max(normalized_dbi_list)
        return {'dbi': dbi_score / k, 'normalized_dbi': normalized_dbi_score / k}
    
    def distance_within_cluster(self, X, element_of_cluster_i):
        sum_of_distance = 0.0
        for i in element_of_cluster_i:
            for j in element_of_cluster_i:
                if i == j:
                    continue
                x_i = X[i]
                x_j = X[j]
                distance_ij = self.Euc_distance(x_i, x_j)
                sum_of_distance = sum_of_distance + distance_ij
        if (len(element_of_cluster_i) * (len(element_of_cluster_i) - 1)) != 0:
            avg_sum_of_distance = sum_of_distance / (len(element_of_cluster_i) * (len(element_of_cluster_i) - 1))
        else:
            avg_sum_of_distance = sum_of_distance
        return avg_sum_of_distance
    
    def Euc_distance(self, list_1, list_2):
        return (np.matrix((list_1 - list_2)) * np.matrix((list_1 - list_2)).T)[0,0]
    
    def dunn(self, X, y, k):
        cluster_element_dict = {}
        for i in range(k):
            cluster_element_dict[i] = []
        
        for i in range(len(y)):
            cluster_index = y[i]
            cluster_element_dict[cluster_index].append(i)
            
        max_distance = []   
        for i in range(k):
            max_distance.append(self.max_distance_within_cluster(X, cluster_element_dict[i]))
        max_distance.append(0.0)
        denominator = max(max_distance)

        min_distance = []
        for i in range(k):
            for j in range(i + 1, k):
                min_distance.append(self.min_distance_across_clusters(X, cluster_element_dict[i], cluster_element_dict[j]))
        numerator = min(min_distance)
        
        if denominator != 0:
            return numerator / denominator
        else:
            return 0.0
        
    def min_distance_across_clusters(self, X, element_of_cluster_i, element_of_cluster_j):
        distance_list = []
        for i in element_of_cluster_i:
            for j in element_of_cluster_j:
                if i == j:
                    continue
                x_i = X[i]
                x_j = X[j]
                distance_ij = self.Euc_distance(x_i, x_j)
                distance_list.append(distance_ij)
        if distance_list == []:
            return 0.0
        else:
            return min(distance_list)
    
    
    def max_distance_within_cluster(self, X, element_of_cluster_i):
        distance_list = []
        for i in element_of_cluster_i:
            for j in element_of_cluster_i:
                if i == j:
                    continue
                x_i = X[i]
                x_j = X[j]
                distance_ij = self.Euc_distance(x_i, x_j)
                distance_list.append(distance_ij)
        distance_list.append(0.0)
        return max(distance_list)
    
    def H(self, result, K):
        cluster_element_dict = {}
        for k in range(K):
            cluster_element_dict[k] = []
            
        for index in range(len(result)):
            cluster_element_dict[result[index]].append(index)
        
        total_length_of_result = 0.0
        for i in cluster_element_dict:
            total_length_of_result = total_length_of_result + len(cluster_element_dict[i])
        H_result = 0.0
        
        for i in cluster_element_dict:
            P_i = float(len(cluster_element_dict[i]))/total_length_of_result
            if P_i == 0.0:
                P_i = 1.0 / total_length_of_result
            H_result = H_result - P_i * (math.log(P_i) / math.log(2))
        return H_result
    
    def evaluate(self, result):
        final_result = {}
        
        print '------------------------------------------------'
        print 'start internal evaluate'
        
        #----------------------- entropy ----------------------
        H = self.H(result, self.K)
        final_result['entropy'] = H
        print H
        
        #----------------------- density -----------------------
        print 'density'
        density_score = self.density(result, self.social_matrix, self.K)
        final_result['density'] = density_score
        print density_score
        
        #------------------------ silhouette ------------------------------
        print 'silhouette'
        if sum(np.array(result)) == 0:
            result[-1] = 1
        print result
        silhouette = self.Silhouette_Coefficient(self.social_matrix, np.array(result), 'euclidean')
        final_result['silhouette'] = silhouette
        print silhouette
        
        #------------------------ dbi ----------------------------
        print 'dbi'
        dbi = self.dbi(self.social_matrix, result, self.K)
        final_result['dbi'] = dbi['dbi']
        final_result['normalized_dbi'] = dbi['normalized_dbi']
        print dbi
        
        #----------------------- Dunn ---------------------------
        #print 'dunn'
        #dunn = self.dunn(self.social_matrix, result, self.K)
        #final_result['dunn'] = dunn
        #print dunn
        
        print 'end internal evaluate'
        print '------------------------------------------------'
        print self.source_target
        print self.type
        return final_result
    
    
    
    