#Polynomial Linear Regression

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


#Fitting Linear Regression to dataset
lin_reg = lm(formula = Salary ~ Level, 
             data = dataset) 
summary(lin_reg)
#Fitting Polynomial Regression to dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ . , data=dataset)
summary(poly_reg)

#Visualize Linear Regression Results

library(ggplot2)
ggplot()+geom_point(aes(x=dataset$Level,y=dataset$Salary),color="red")+
  geom_line(aes(x=dataset$Level,y=predict(lin_reg)),color="blue")+
  ggtitle("Truth or Bluff(Linear Regression)")+
  xlab("Level")+
  ylab("Salary")
#Visualize Polynomial Regression Results

  ggplot()+geom_point(aes(x=dataset$Level,y=dataset$Salary),color="red")+
    geom_line(aes(x=dataset$Level,y=predict(poly_reg)),color="blue")+
    ggtitle("Truth or Bluff(Polynomial Regression)")+
    xlab("Level")+
    ylab("Salary")
  
#Visualize Polynomial Regression Results (For high resolution and smoother curve)
  X_grid = seq(min(dataset$Level),max(dataset$Level), 0.1)
  ggplot()+geom_point(aes(x=dataset$Level,y=dataset$Salary),color="red")+
    geom_line(aes(x=X_grid, y=predict(poly_reg, newdata = data.frame(Level = X_grid, Level2 =X_grid^2,Level3=X_grid^3,Level4=X_grid^4))),color="blue")+
    ggtitle("Truth or Bluff(Polynomial Regression)")+
    xlab("Level")+
    ylab("Salary")

  
#Predicting a new Result with Linear Regression
y_pred = predict(lin_reg, data.frame(Level=6.5))

#Predicting a new result with Polynomial regression
predict(poly_reg,data.frame(Level = 6.5,Level2=6.5^2, Level3=6.5^3, Level4=6.5^4))


