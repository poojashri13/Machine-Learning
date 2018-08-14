#Multiple Linear Regression

#Importing Dataset
dataset = read.csv('50_Startups.csv')
# dataset = dataset[,2:3]

#Encoding Categorical data
dataset$State=factor(dataset$State,
                     levels=c('New York','California','Florida'),
                     labels = c(1,2,3))


#Splitting dataset into Training set and test set
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit,SplitRatio = 0.8) #put split ratio for training set
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# #Feature Scaling
# 
# training_set[,2:3] = scale(training_set[,2:3]) #country and purchased is not numeric these are factors
# test_set[,2:3] = scale(test_set[,2:3])


#Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = Profit ~ ., data=training_set ) # instead of writing all the independent variable write. R.D.Apend + Administration + Marketing.Spend + State -> .


#Predicting the test set result

y_pred = predict(regressor, newdata = test_set)

#Building the optimal model using backward elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data=dataset)
summary(regressor)


regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend , data=dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend  + Marketing.Spend , data=dataset)
summary(regressor)


regressor = lm(formula = Profit ~ R.D.Spend , data=dataset)
summary(regressor)