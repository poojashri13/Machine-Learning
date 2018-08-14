# Hierachical clustering

dataset = read.csv("Mall_Customers.csv")
X = dataset[4:5]


#Using Dendogram to find Optimal number of cluster

dendogram = hclust(dist(X, method = "euclidean"),method = "ward.D")
plot(dendogram, main = "Dendogram", xlab = "Customers",ylab = "Euclidean Distances")

#Fitting Hierchical clusterint on mall dataset
hc = hclust(dist(X, method = "euclidean"),method = "ward.D")
y_hc = cutree(hc, 5)

#Visualizing the clusters
library(cluster)
clusplot(X, 
         y_hc,
         lines = 0,
         shade = T,
         color =T,
         labels=2,
         plotchar = F,
         span=T,
         main="Clusters of clients",
         xlab = "Annual Income",
         ylab = "Spending Score")