#Importing Dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[,2:3]

# Splitting dataset into Training set and test set
# install.packages("caTools")
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased,SplitRatio = 0.8) #put split ratio for training set
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set[,2:3] = scale(training_set[,2:3]) #country and purchased is not numeric these are factors
# test_set[,2:3] = scale(test_set[,2:3])


#FitRegression model to dataset
library(rpart)
regressor = rpart(formula = Salary ~.,data=dataset,control = rpart.control(minsplit = 1))

#Predicting a new result with Regression Model
y_pred = predict(regressor,data.frame(Level = 6.5))


#visualize Regression Results

library(ggplot2)
ggplot()+geom_point(aes(x=dataset$Level,y=dataset$Salary),color="red")+
  geom_line(aes(x=dataset$Level,y=predict(regressor,newdata=dataset)),color="blue")+
  ggtitle("Truth or Bluff( Decision Tree Regression model)")+
  xlab("Level")+
  ylab("Salary")

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression Model)') +
  xlab('Level') +
  ylab('Salary')
