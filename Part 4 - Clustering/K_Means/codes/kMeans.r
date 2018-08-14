#K-means clustering


#importing mall dataset
dataset = read.csv("MAll_Customers.csv")
X = dataset[,4:5]

# Using the Elbow Method to find the optimal number of clusters
set.seed(6)
wcss = vector()
for(i in 1:10){
  wcss[i] = sum(kmeans(X,i)$withinss)
}
plot(1:10, wcss, type = "b", main = "Clusters of clients", xlab = "Number of clusters", ylab = "WCSS")

#Applying kMeans to the Mall customers
set.seed(29)
kmeans = kmeans (X, 5, iter.max = 300,nstart = 10)

#Visualizing the clusters
clusplot(X, 
         kmeans$cluster,
         lines = 0,
         shade = T,
         color =T,
         labels=2,
         plotchar = F,
         span=T,
         main="Clusters of clients",
         xlab = "Annual Income",
         ylab = "Spending Score")