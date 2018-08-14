# Data PreProcessing

#Importing Dataset
dataset = read.csv('Data.csv')

#Taing care of Missing Values
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN=function(x)mean(x,na.rm=TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary), ave(dataset$Salary, FUN=function(x)mean(x,na.rm=TRUE)),dataset$Salary)
?ave


#Encoding Categorical data
dataset$Country=factor(dataset$Country,levels=c('France','Spain','Germany'),labels = c(1,2,3))
dataset$Purchased=factor(dataset$Purchased,levels=c('Yes','No'),labels = c(1,0))


#Splitting dataset into Training set and test set
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio = 0.8) #put split ratio for training set
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling

training_set[,2:3] = scale(training_set[,2:3]) #country and purchased is not numeric these are factors
test_set[,2:3] = scale(test_set[,2:3])