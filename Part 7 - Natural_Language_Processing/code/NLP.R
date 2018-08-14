#importing Dataset 
dataset_original = read.delim("Restaurant_Reviews.tsv",quote = '', stringsAsFactors = F)
summary(dataset)


#Cleaning text
library(SnowballC)
library(tm)
corpus = VCorpus(VectorSource(dataset$Review))
corpus = tm_map(corpus,content_transformer(tolower))
corpus = tm_map(corpus,removeNumbers)
corpus = tm_map(corpus,removePunctuation)
corpus = tm_map(corpus,removeWords, stopwords())
corpus = tm_map(corpus,stemDocument)
corpus = tm_map(corpus, stripWhitespace)

#Creating Bag of Words model

dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dtm
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked


#Encoding

dataset$Liked = factor(dataset$Liked, levels=c(0,1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-2] = scale(training_set[-2])
test_set[-2] = scale(test_set[-2])

# Fitting SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = naiveBayes(x = training_set[-2],
                        y = training_set$Purchased)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-2])

# Making the Confusion Matrix
cm = table(test_set[, 2], y_pred)
