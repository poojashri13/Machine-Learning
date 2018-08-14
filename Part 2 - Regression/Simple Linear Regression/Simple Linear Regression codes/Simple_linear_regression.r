# Data PreProcessing

#Importing Dataset
dataset = read.csv('Salary_Data.csv')
# dataset = dataset[,2:3]

#Splitting dataset into Training set and test set
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary,SplitRatio = 2/3) #put split ratio for training set
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# #Feature Scaling
# 
# training_set[,2:3] = scale(training_set[,2:3]) #country and purchased is not numeric these are factors
# test_set[,2:3] = scale(test_set[,2:3])

#Simple Linear Regression

regressor = lm(formula = Salary ~ YearsExperience , data = training_set)

# summary(regressor)
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)        25592       2646   9.672 1.49e-08 *** 
#   YearsExperience     9365        421  22.245 1.52e-14 *** -> high statistical significance
# 0 *-> no statistical significance
# ***->high statistical significance

y_pred <- predict(regressor,newdata = test_set)

#Visualising the Training set result
library(ggplot2)

ggplot()+
  geom_point(aes(x=training_set$YearsExperience,y=training_set$Salary),
             color="red") +
  geom_line(aes(x=training_set$YearsExperience, y = predict(regressor,newdata = training_set)),color="blue")+
  ggtitle('Salary vs Experience(Training set)')+xlab("Years of Experience")+ylab("salary")
  

ggplot()+
  geom_point(aes(x=test_set$YearsExperience,y=test_set$Salary), color="red")+
  geom_line(aes(training_set$YearsExperience,y=predict(regressor,newdata = training_set)),color="blue")+
  ggtitle('Salary vs Experience(Test set)')+xlab("Years of Experience")+ylab("salary")


