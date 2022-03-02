#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
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
from itertools import count
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from tqdm import tqdm

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#




def PCA(points):

    points_mean = points.mean(axis=0)
    centered = (points - points_mean)[:,:,None]
    # stacking matrices in the first diemension
    cov = (np.matmul(centered, centered.transpose(0,2,1))).mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvalues, eigenvectors

def compute_local_PCA(query_points, cloud_points, radius):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    kdtree = KDTree(cloud_points)

    neighborhoods = kdtree.query_radius(query_points, radius)

    all_eigenvalues = np.zeros((cloud.shape[0], 3))
    all_eigenvectors = np.zeros((cloud.shape[0], 3, 3))
    for i, idx in enumerate(neighborhoods):
        val, vec = PCA(cloud_points[idx,:])
        all_eigenvalues[i] = val
        all_eigenvectors[i] = vec
    return all_eigenvalues, all_eigenvectors

def compute_local_PCA_knn(query_points, cloud_points, k):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    kdtree = KDTree(cloud_points, leaf_size=k*2)

    _, neighborhoods = kdtree.query(query_points, k)

    all_eigenvalues = np.zeros((cloud.shape[0], 3))
    all_eigenvectors = np.zeros((cloud.shape[0], 3, 3))
    for i, idx in tqdm(enumerate(neighborhoods)):
        val, vec = PCA(cloud_points[idx,:])
        all_eigenvalues[i] = val
        all_eigenvectors[i] = vec
    return all_eigenvalues, all_eigenvectors

def compute_features(query_points, cloud_points, radius):

    eigenvalues, eigenvectors = compute_local_PCA(query_points, cloud_points, radius)

    eigenvalues[:,2] += 1e-10
    
    linearity = 1 - eigenvalues[:,1]/eigenvalues[:,2]
    planarity = (eigenvalues[:,1] - eigenvalues[:,0]) / eigenvalues[:,2]
    sphericity = eigenvalues[:,0] / eigenvalues[:,2]

    ez = np.zeros((3,1))
    ez[2,0] = 1.

    n = eigenvectors[:,:,0]
    verticality = 2 * np.arcsin(np.abs(n@ez)) / np.pi

    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        print(cloud.shape)
        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

		
    # Normal computation
    # ******************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, 0.50)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../Lille_street_small_normals.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
		
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA_knn(cloud, cloud, 5)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../Lille_street_small_normals_knn.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
		

    # feature computation
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        verticality, linearity, planarity, sphericity = compute_features(cloud, cloud, 0.5)

        write_ply('../Lille_street_small_feat.ply', [cloud, verticality, linearity, planarity, sphericity], ['x', 'y', 'z', 'vv', 'll', 'pp', 'ss'])
