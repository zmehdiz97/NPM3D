#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):

    neighborhoods = []
    for query in queries: 
        query = query[None, :]
        distance = cdist(query, supports,'sqeuclidean').reshape(-1)
        indices = np.where(distance < radius)[0]
        neighborhoods.append(supports[indices])
    return neighborhoods


def brute_force_KNN(queries, supports, k):

    neighborhoods = []
    for query in queries: 
        query = query[None, :]
        distance = cdist(query, supports,'sqeuclidean').reshape(-1)
        indices = np.argpartition(distance, k)[:k]
        neighborhoods.append(supports[indices])
    return neighborhoods


def hierarchical_spherical(queries, supports, radius, leaf_size=2):

    tree = KDTree(supports, leaf_size = leaf_size)
    neighborhoods = tree.query_radius(queries, radius)

    return neighborhoods

def plot_exec(x, y, x_label='Radius', semilogx=False):

    plt.figure(figsize = (6, 6))
    if semilogx:
        plt.semilogx(x, y)
    else:
        plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel('Computing time')
    plt.grid()
    plt.title(f'Computing time of spherical neighborhoods unsing KDTree with respect to {x_label}')
    plt.show()



# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if True:

        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

 



    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if True:

        # Define the search parameters
        num_queries = 1000
        #leaf_size = [2,4,6,10,14,18,25,30,40,50,60,80,100,120,150,200,250,1e3,1e4, 5e4, 1e5, 5e5,1e6]
        leaf_size = 1e4
        radius = np.linspace(0.1, 2.0, num=20)
        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        #y=[]
        #for r in radius:
        #    t0 = time.time()
        #    neighborhoods = hierarchical_spherical(queries, points, r, leaf_size= leaf_size)
        #    t1 = time.time()
        #    y.append(t1-t0)
#
        #plot_exec(radius, y, x_label='radius', semilogx=False)
        t0 = time.time()
        neighborhoods = hierarchical_spherical(queries, points, 0.2, leaf_size= leaf_size)
        s=sum([len(n) for n in neighborhoods])
        print(s / len(neighborhoods))
        t1 = time.time()
        # Print timing results
        print('{:d} KDTree spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        print('Computing KDTree spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        
        
        
        