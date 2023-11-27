### K means
given a graph that has potential points that tend to "cluster", the algorithm randomly assigns a position of more than one "centroid" and then calculates the distance of each point of the graph to all centroids. the points closest to each centroid are now labelled as belonging to that centroid and then the centroid is moved to the average position of all the positions that belong to it and then the process starts again, reclassify, move to average position, rinse and repeat until there are no changes in position owner ship, then you should have a centroid in the middle of each cluster

### optimisation
the cost function is actually just the algorithm itself, it aims to reduce the average distance to the points from the respective centroid
![[Pasted image 20231119112334.png]]

### initialising cluster centroid

a good way is to randomly pick training examples and init the centroids to that same position

Sometimes you might end up with whack clustering due to unfortunate placing of the cluster centroids and in that case you might want to run the algorithm multiple times with different start points and then compute the cost for each and choose the one with the lowest cost
![[Pasted image 20231119114025.png]]

### choosing the number of clusters

the right value of K is ambiguous ... buutt

Elbow method:
- run the cost function on many values of clusters and choose the value of k where the cost function decreases not so rapidly at some number of Ks

A better approach might be to choose a value of K that fits the use case of the data better like doing some analysis on t-shirt sizes, there are only a couple sizes in men's cloths so having 19 clusters wouldn't make much sense, instead opt for 3-5 for sm-md-lg-xl-xxl

