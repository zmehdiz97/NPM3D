#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

 
# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply
from visu import show_ICP

import sys


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # YOUR CODE
    #R = np.eye(data.shape[0])
    #T = np.zeros((data.shape[0],1))

    data_mean = data.mean(axis=1, keepdims=True)
    ref_mean = ref.mean(axis=1, keepdims=True)
    
    Q_data = data - data_mean
    Q_ref = ref - ref_mean
    H = Q_data @ Q_ref.T

    U, S, V = np.linalg.svd(H)

    R = V.T @ U.T
    if np.linalg.det(R) < 0:
        U[:,-1] *= -1
        R = V.T @ U.T

    T = ref_mean - R @ data_mean

    return R, T



def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''
    RMS = lambda data_, ref_: np.sqrt( np.mean( np.sum(np.power(data_ - ref_, 2), axis=0) ) )

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    d, n = data.shape
    R_list = [np.eye(d)]
    T_list = [np.zeros((d,1))]
    neighbors_list = []
    RMS_list = []

    kdtree = KDTree(ref.T)
    for i in range(max_iter):
        neighbors = kdtree.query(data_aligned.T, k=1, return_distance = False)
        neighbors = neighbors.squeeze()
        neighbors_list.append(neighbors)

        R, T = best_rigid_transform(data_aligned, ref[:,neighbors])
        data_aligned = R @ data_aligned + T

        T_list.append(R @ T_list[-1] + T)
        R_list.append(R @ R_list[-1])
    
        rms = RMS(data_aligned, ref[:,neighbors])
        RMS_list.append(rms)

        if rms < RMS_threshold:
            return data_aligned, R_list[1:], T_list[1:], neighbors_list, RMS_list
    
    return data_aligned, R_list[1:], T_list[1:], neighbors_list, RMS_list

def icp_point_to_point_fast(data, ref, max_iter, RMS_threshold, n_samples):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''
    RMS = lambda data_, ref_: np.sqrt( np.mean( np.sum(np.power(data_ - ref_, 2), axis=0) ) )

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    d, n = data.shape
    R_list = [np.eye(d)]
    T_list = [np.zeros((d,1))]
    neighbors_list = []
    RMS_list = []

    kdtree = KDTree(ref.T)

    for i in range(max_iter):
        rand_idx = np.random.choice(n, n_samples, replace=False)
        neighbors = kdtree.query(data_aligned[:,rand_idx].T, k=1, return_distance = False)
        neighbors = neighbors.squeeze()
        R, T = best_rigid_transform(data_aligned[:,rand_idx], ref[:,neighbors])
        data_aligned = R @ data_aligned + T

        T_list.append(R @ T_list[-1] + T)
        R_list.append(R @ R_list[-1])
    
        rms = RMS(data_aligned[:,rand_idx], ref[:,neighbors])
        RMS_list.append(rms)

        if rms < RMS_threshold:
            return data_aligned, R_list[1:], T_list[1:], neighbors_list, RMS_list
    
    return data_aligned, R_list[1:], T_list[1:], neighbors_list, RMS_list

#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if True:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

		# Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        bunny_r_opt = R.dot(bunny_r) + T

        # Save cloud
        write_ply('../bunny_r_opt', [bunny_r_opt.T], ['x', 'y', 'z'])

        # Compute RMS
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))
        print(RMS_after)


    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'
        
        # Load clouds
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y']))        

        # Apply ICP
        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data2D, ref2D, 10, 1e-4)
        
        # Show ICP
        show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.show()
        

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'
        
        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        # Apply ICP
        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)
        
        # Show ICP
        show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.show()

    # If statement to skip this part if wanted
    if True:

        # Cloud paths
        bunny_o_path = '../data/Notre_Dame_Des_Champs_1.ply' 
        bunny_p_path = '../data/Notre_Dame_Des_Champs_2.ply'
        
        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))
        n_samples = [1000, 10000, 50000, 100000]
        rms_list = []
        for n in n_samples:
        # Apply ICP
            bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point_fast(bunny_p, bunny_o, 100, 1e-4, 10000)
            rms_list.append(RMS_list)
        #write_ply('../Notre_Dame_Des_Champs_ICP', [bunny_p_opt.T], ['x', 'y', 'z'])
        # Show ICP
        #show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        for sl, rms in zip(n_samples, rms_list):
            plt.plot(rms, label="{} samples".format(sl))
        plt.title("Evolution of the RMS at each step of fast ICP")
        plt.legend()
        plt.show()
