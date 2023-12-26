import numpy as np
import matplotlib.pyplot as plt
from utils import *
get_ipython().run_line_magic('matplotlib', 'inline')
def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """
    # Set K
    K = centroids.shape[0]
    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)
    ### START CODE HERE ###
    for i in range(X.shape[0]):
        distance = np.empty(centroids.shape[0])
        for j in range(centroids.shape[0]):
            norm_ij = np.sqrt(np.sum((X[i] - centroids[j])**2))
            distance[j] = norm_ij
        idx[i] = np.argmin(distance)
     ### END CODE HERE ###
    
    return idx
X = load_data()
print("First five elements of X are:\n", X[:5]) 
print('The shape of X is:', X.shape)
initial_centroids = np.array([[3,3], [6,2], [8,5]])
idx = find_closest_centroids(X, initial_centroids)
print("First three elements in idx are:", idx[:3])
from public_tests import *
find_closest_centroids_test(find_closest_centroids)
def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    
    ### START CODE HERE ###
    for k in range(K):
        points = X[idx == k]
        centroids[k] = np.mean(points, 0)
    ### END CODE HERE ## 
    
    return centroids
K = 3
centroids = compute_centroids(X, idx, K)
print("The centroids are:", centroids)
compute_centroids_test(compute_centroids)
def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))
    # Run K-Means
    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx
X = load_data()
initial_centroids = np.array([[3,3],[6,2],[8,5]])
max_iters = 10
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)
def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray): Initialized centroids
    """
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids
K = 3
max_iters = 10
initial_centroids = kMeans_init_centroids(X, K)
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)
original_img = plt.imread('bird_small.png')
plt.imshow(original_img)
print("Shape of original_img is:", original_img.shape)
X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))
K = 16
max_iters = 10
initial_centroids = kMeans_init_centroids(X_img, K)
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)
print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])
plot_kMeans_RGB(X_img, centroids, idx, K)
show_centroid_colors(centroids)
idx = find_closest_centroids(X_img, centroids)
X_recovered = centroids[idx, :] 
X_recovered = np.reshape(X_recovered, original_img.shape) 
fig, ax = plt.subplots(1,2, figsize=(16,16))
plt.axis('off')
ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].set_axis_off()
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()
