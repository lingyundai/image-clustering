#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import re
from scipy.sparse import csr_matrix
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from collections import defaultdict
from sklearn.feature_selection import VarianceThreshold


# In[2]:


# import data
image_data = pd.read_csv('test-data-images.txt', header=None)
image_data


# In[3]:


# parse the data points from str to float
class RecordParser:
    def __init__(self):
        pass
    
    def parser(self, arr):
        # parse the data to float type
        data_float = []
        
        arr = arr.to_numpy()

        for row in arr:
            nums = re.findall(r'[0-9.]+', str(row))
            res = []
            for num in nums:
                num = float(num)
                res.append(num)
            data_float.append(res)
        
        return data_float

obj = RecordParser()
image_data_parsed = obj.parser(image_data)


# In[4]:


# scaler = StandardScaler()
# image_data = scaler.fit_transform(image_data_parsed)
# image_data


# In[5]:


image_data = csr_matrix(image_data_parsed)
image_data = TSNE(n_components=1, learning_rate='auto', init='random', 
                  perplexity=420, random_state=13).fit_transform(image_data)
image_data


# In[6]:


class KMeans:
    def __init__(self):
        pass
    
    # compute euclidean distance
    def computeEuclideanDistance(self, points, centroid):
        euclidean_distance = np.sqrt(sum([np.square(data_points - centroid_item) 
                                          for data_points, centroid_item in zip(points, centroid)]))
        
#         print("euclidean_distance: ", euclidean_distance)
        return euclidean_distance
    
    # for each cluster, compute datapoints to the cluster centroid distances and
    # store in cluster_distance_map
    def computeWithinClusterDistances(self, data, centroids):
        # to store cluter's all data point to centroid distances 
        cluster_distances_map = {}
                
        for centroid in centroids:
            cluster_distances_map[tuple(centroid)] = []
            
        for i, points in enumerate(data):
            distances = []
            for j, centroid in enumerate(centroids):
#                 print("data: ", points)
#                 print("centroid: ", centroid)
                euclidean_distance = self.computeEuclideanDistance(points, centroid)

                distances.append(euclidean_distance)
                
            # find out the index of the min distance in the arr, which 
            # determines the corresponding centroid 
            min_dist_index = distances.index(min(distances))
#             print("dist arr: ", distances)
#             print("min_dist_index: ", min_dist_index)
        
            # distances from the data points to each centroid are added to cluster distances map
            cluster_distances_map[tuple(centroids[min_dist_index])].append(distances[min_dist_index])
        
#         print("cluster_distances_map: ", cluster_distances_map)
        return cluster_distances_map
    
    # for each cluster, store the datapoints in that cluster to clusters map
    def collectClusterDataPoints(self, data, centroids):
        clusters = {}
        
        for centroid in centroids:
            clusters[tuple(centroid)] = []
            
        for i, points in enumerate(data):
#             print("data: ", points)
            distances = []
            for j, centroid in enumerate(centroids):
#                 print("centroid: ", centroid)
                euclidean_distance = self.computeEuclideanDistance(points, centroid)

                distances.append(euclidean_distance)
                
            # find out the index of the min distance in the arr, which 
            # determines the corresponding centroid 
            min_dist_index = distances.index(min(distances))
#             print("dist arr: ", distances)
#             print("min_dist_index: ", min_dist_index)

            clusters[tuple(centroids[min_dist_index])].append(data[i])
        
#         print("clusters: ", clusters)
        return clusters
        
    # for each cluster, compute individual datapoints SSE and store in
    # cluster_sse_map
    def computeClusterSSE(self, cluster_distances_map):
        cluster_sse_map = {}
        for k, v in cluster_distances_map.items():
            cluster_data_points = cluster_distances_map[k]
            sse_sum = np.sum(np.square(cluster_data_points))
            cluster_sse_map[k] = sse_sum
            
#         print("cluster_sse_map: ", cluster_sse_map)
        return cluster_sse_map
    
    # for each cluster, sum up the total of SSEs
    def computeTotalSSE(self, sse_map):
        total_sse = 0
        for k,v in sse_map.items():
            total_sse += sse_map[k]
            
#         print("total sse: ", total_sse)
        return total_sse
    
    # compute new centroids for each cluster by the mean of the data points, 
    # take in clusters map which has all data points for corresponding cluster
    def computeNewCentroids(self, clusters):
        new_centroids = []
        
        for k, v in clusters.items():
            # the data points in each cluster
            data_points = clusters[k]
#             print("data_points: ", data_points)

            # new centroid in each cluster is the mean of all points in the cluster
            new_centroid = (np.sum(data_points, axis=0) / len(data_points)).tolist()
            new_centroids.append(new_centroid)
    
#         print("new centroids: ", new_centroids)
        return new_centroids
    
    # compute new centroids for each cluster by k-means++
    def computeKMeansPlusPlusNewCentroids(self, cluster_distances_map, clusters):
        new_centroids = []
        for (centroid_key, data_points), (cluster_key, cluster_data) in zip(cluster_distances_map.items(), clusters.items()):
            data_points = cluster_distances_map[centroid_key]
            data_points_sum_square = np.square((np.sum(data_points)))
            
            probabilities = []
            for item in data_points:
                probability = np.square(item) / data_points_sum_square
                probabilities.append(probability)
            
            data_points_cluster = clusters[cluster_key]
            index = probabilities.index(max(probabilities))
            new_centroid = data_points_cluster[index]
            new_centroids.append(new_centroid)
#         print("newwwww: ", new_centroids)
        return new_centroids
            
    
    def generateClusterAssignmentRes(self, data, clusters):
        # collect the centroids cluster keys
        cluster_keys = list(clusters.keys())
        
        for row in data:
            for k, v in clusters.items():
                # if the row is a value in a cluster
                if row in clusters[k]:
                    # print out the index of the centroid in cluster keys + 1, because
                    # the cluster number starts from one
                    print(cluster_keys.index(k) + 1)
    
    # validate cluster accuracy by computing silhouette coefficient for individual point
    def computeSilhouetteCoefficient(self, clusters):
        # calculate cohesion
        # a = average distance of i to the points in its cluster
        # avg within cluster distances map with the key of
        # each centroid data point and the avg distances of
        # a data point to other data points as values in the cluster.
        avg_within_cluster_distances = defaultdict(list)
        for k, v in clusters.items():
            within_cluster_point_distances = []
            data_points = clusters[k]
            for item in data_points:
                for data_point in data_points:
                    if data_point != item: 
                        # dist from a data point in a cluster to each/another other
                        # data points in the cluster is computed through euclidean distance
                        # the result is appended to within cluster point distances array,
                        # once all distances to other data points are computed, avg distance
                        # is computed based on the sum of the within cluster point distances array
                        # divided by the total length of the array because that means how many
                        # data points we compared.
                        dist = self.computeEuclideanDistance(item, data_point)
                        within_cluster_point_distances.append(dist)
                        
                avg_distance = np.sum(within_cluster_point_distances) / len(within_cluster_point_distances)
                # avg distance from a datapoint to other data points in the cluster
                # is then added to the map of avg within cluster distances, with the 
                # associated centroid data point as key. so that we know the avg distance
                # of this data point belongs to what cluster.  The length of the values
                # in this map is the same amount of the data points in the cluster. 
                # each avg distance can be traced to corresponding data point in the cluster.
                avg_within_cluster_distances[k].append(avg_distance)
                
                # avg_within_cluster_distances is a collection of a in each cluster. 
                # to compute individual point's silhouette coefficient, we will 
                # match the individual a to b in the map below for separation. 
                # and compute the formula. i's position should be the same
                # in these 2 dictionary, because we are computing points sequentially
                # from the same clusters map.
                
        # calculate separation
        # b = min (average distances of i to points in another cluster k,
        # for all k not containing i)
        avg_other_cluster_distances = defaultdict(list)
        
        # collection of the data points from all clusters
        data_points_all_clusters = []
        for k, v in clusters.items():
            data_points = clusters[k]
            data_points_all_clusters.append(data_points)
#         print(data_points_all_clusters)
        
        # to prevent duplicate key issue, every cluster item
        # is mapped to unique index in incrementing order,
        # so that in map we are not going to see a key asscoiated
        # with more values we expected
        data_point_index_map = {}
        index_count = 0
        for data in data_points_all_clusters:
            for item in data:
                data_point_index_map[tuple(item)] = index_count
                index_count += 1
#         print("ah: ", data_point_index_map)
        
        # for cluster data points
        avg_distances_map = defaultdict(list)
        dists_to_other_data_points = []
        distances_map = defaultdict(list)
        for i, cluster_data_point_i in enumerate(data_points_all_clusters):
            # now calculate each data point in a cluster's distance to 
            # all other data points in another cluster
            for j, cluster_data_point_j in enumerate(data_points_all_clusters):
                if i != j:
#                     print(i, j)
                    # now we have access to each data point in a cluster against each
                    # data point in another cluster, we calculate the euclidean distance
                    # from the data point to each data point in another cluster and get
                    # the avg, which represents the avg separation for this specific
                    # data point to all data points in another cluster
                    for cluster_data_point in cluster_data_point_i:
                        for item in cluster_data_point_j:

                            dist = self.computeEuclideanDistance(cluster_data_point, item)
                            # append to a distance map with associated index number of 
                            # the key (the associated key that is being
                            # computed against other data points), so it is easier to gather
                            # the avg separation for each data point without duplicate
                            # keys generates incorrect key distance mapping
                            index_to_map = data_point_index_map[tuple(cluster_data_point)]
                            distances_map[index_to_map].append(dist)
                        
                        # this step is (avg dist(c1, c2), avg dist(c1, c3)) in separation
                        dists = distances_map[index_to_map]
                        avg_dist = np.sum(dists) / len(dists)
                        avg_distances_map[index_to_map].append(avg_dist)
                        distances_map[index_to_map] = []
        
#         print("clusters: ", clusters)
#         print("avg_distances_map: ", avg_distances_map)
    
        # now we need to find the min of the avg distances for each
        # data point to other cluster data points. 
        min_avg_distances = defaultdict(list)
        for k, v in avg_distances_map.items():
            vals = avg_distances_map[k]
#             print(vals)
            min_avg_distances[k].append(min(vals))
#         print("min_avg_distances: ", min_avg_distances)
        
        # now we append the min separation for each data point
        # to the corresponding index in the corresponding centroid
        separations_in_clusters = defaultdict(list)
        for k, v in clusters.items():
            vals = clusters[k]
            for item in vals:
                min_dist_item = min_avg_distances[data_point_index_map[tuple(item)]]
#                 print(min_dist_item)
                separations_in_clusters[k].append(min_dist_item)
#         print(separations_in_clusters)

        # calculate silhouette coefficient for each data point
        silhouette_coefficient_data_points = []
        points_cohesion = np.concatenate([avg_within_cluster_distances[k] for k, 
                                          v in avg_within_cluster_distances.items()]).tolist()
        points_separation = np.concatenate([separations_in_clusters[k] for k, 
                                            v in separations_in_clusters.items()]).tolist()
        
#         print(points_cohesion)
#         print(points_separation)
        
        for i, a in enumerate(points_cohesion):
            b = points_separation[i][0]
            sc = (b - a) / max(a, b)
            silhouette_coefficient_data_points.append(sc)
#         print(silhouette_coefficient_data_points)
        
        # now we have the sc for all data points, we can compute avg_silhouette coefficient
        avg_silhouette_coefficient = np.sum(silhouette_coefficient_data_points) / len(silhouette_coefficient_data_points)
        print("average silhouette coefficient: ", avg_silhouette_coefficient)
            
                
    def runKMeans(self, data, centroids):
        total_sse_history = []
        # keep executing the methods until stop condition is met
        while True:
            cluster_distances_map = self.computeWithinClusterDistances(data, centroids)
            clusters = self.collectClusterDataPoints(data, centroids)
            cluster_sse = self.computeClusterSSE(cluster_distances_map)
            total_sse = self.computeTotalSSE(cluster_sse)
            total_sse_history.append(total_sse)
            new_centroids = self.computeNewCentroids(clusters)
#             new_centroids = self.computeKMeansPlusPlusNewCentroids(cluster_distances_map, clusters)
            # stop if new_centroids is the same as prev centroids, centroids stop updating
            if new_centroids == centroids:
                print("Total Cluster SSEs: ", total_sse)
                self.computeSilhouetteCoefficient(clusters)
                self.generateClusterAssignmentRes(data, clusters)
                # plot the SSE history to observe the SSE change because
                # the goal is to minimize SSE
                plt.plot(range(1, len(total_sse_history) + 1), total_sse_history)
                plt.title('Clusters SSE Sum in Iterations')
                plt.xlabel('Iteration')
                plt.ylabel('Total SSE')
                plt.grid(True)
                plt.show()
                print("The cluster centroids stopped updating, stop condition met.")
                break
            
            centroids = new_centroids
    
    # plot number of clusters k and final total SSE for each k,
    # will be called for the best performing experiments
    def runKMeansKClusters(self, data):
        cluster_final_total_sse = []
        
        for k in range(2, 21, 2):
            random_initial_centroids = random.sample(data, k)
            
            centroids = random_initial_centroids
            total_sse_history = []
        
            while True:
                cluster_distances_map = self.computeWithinClusterDistances(data, centroids)
                clusters = self.collectClusterDataPoints(data, centroids)
                cluster_sse = self.computeClusterSSE(cluster_distances_map)
                
                # total_sse history
                total_sse = self.computeTotalSSE(cluster_sse)
                total_sse_history.append(total_sse)
                
                new_centroids = self.computeNewCentroids(clusters)
    #             new_centroids = self.computeKMeansPlusPlusNewCentroids(cluster_distances_map, clusters)
                # stop if new_centroids is the same as prev centroids, centroids stop updating
                if new_centroids == centroids:
                    # the last sse in the sse history is the final sse
                    cluster_final_total_sse.append(total_sse_history[-1])
                    print("Total Cluster SSEs for {} clusters: {}".format(k, total_sse))
                    break
                
                centroids = new_centroids
            
        # plot the SSE history to observe the SSE change because
        # the goal is to minimize SSE
        plt.plot(range(1, len(cluster_final_total_sse) + 1), cluster_final_total_sse)
        plt.title('Total SSEs in Number of Clusters K')
        plt.xlabel('Number of Clusters K')
        plt.ylabel('Total SSE')
        plt.grid(True)
        plt.show()
            
np.random.seed(88)

# toy_data = [[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4], [9, 10, 11, 12], [-3, -4, -5, -6], [0, 1, 2, 3]]
# random_initial_centroids = [[5, 6, 7, 8], [-1, -2, -3, -4], [0, 1, 2, 3]]

random_initial_centroids = random.sample(image_data.tolist(), k=10)
print("random_initial_centroids: ", random_initial_centroids)

obj = KMeans()
obj.runKMeans(image_data.tolist(), random_initial_centroids)


# In[ ]:


# generates "plot of total cluster SSEs vs. value of cluster K increasing from 2 to 20 in steps of 2 (x-axis)"
obj.runKMeansKClusters(image_data.tolist())

