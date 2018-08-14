# Regression Model


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
#Create our Regressor model


#Predicting a new result with Regression Model
y_pred = predict(regressor,data.frame(Level = 6.5))

#Visualize Regression Results
library(ggplot2)

ggplot()+geom_point(aes(x=dataset$Level,y=dataset$Salary),color="red")+
  geom_line(aes(x=dataset$Level,y=predict(poly_reg)),color="blue")+
  ggtitle("Truth or Bluff(Regression model)")+
  xlab("Level")+
  ylab("Salary")

#Additional Trick


#Visualize Regression Results (For high resolution and smoother curve)
library(ggplot2)
X_grid = seq(min(dataset$Level),max(dataset$Level), 0.1)
ggplot()+geom_point(aes(x=dataset$Level,y=dataset$Salary),color="red")+
  geom_line(aes(x=X_grid,y=predict(regressor,newdata=data.frame(Level = X_grid))),color="blue")+
  ggtitle("Truth or Bluff(Regression model)")+
  xlab("Level")+
  ylab("Salary")






