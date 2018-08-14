#Apriori Algorithm


#Data PreProcessing
#install.packages("arules")
library(arulesViz)
library(arule)
dataset = read.csv("Market_Basket_Optimisation.csv", header=F)
dataset = read.transactions("Market_Basket_Optimisation.csv", sep=",", rm.duplicates = TRUE)
image(dataset)
summary(dataset)
#Frequency plot
itemFrequencyPlot(dataset, topN=10)

#Train Apriori on the dataset


#Min. Support = assume we have bought a product 3 times a day so in a week we have bought 3*7 times so support = 3}*7/7500
#confidence: take a random min value as confidence
rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.8)) #no rules created
rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.4))

plot(rules)
#Visualizing the results
inspect(sort(rules, by="lift")[1:10])
# lhs                                            rhs                 support     confidence lift    
# [1]  {mineral water,whole wheat pasta}           => {olive oil}         0.003866151 0.4027778  6.115863
# [2]  {spaghetti,tomato sauce}                    => {ground beef}       0.003066258 0.4893617  4.980600
# [3]  {french fries,herb & pepper}                => {ground beef}       0.003199573 0.4615385  4.697422
# [4]  {cereals,spaghetti}                         => {ground beef}       0.003066258 0.4600000  4.681764
# [5]  {frozen vegetables,mineral water,soup}      => {milk}              0.003066258 0.6052632  4.670863
# [6]  {chocolate,herb & pepper}                   => {ground beef}       0.003999467 0.4411765  4.490183
# [7]  {chocolate,mineral water,shrimp}            => {frozen vegetables} 0.003199573 0.4210526  4.417225
# [8]  {frozen vegetables,mineral water,olive oil} => {milk}              0.003332889 0.5102041  3.937285
# [9]  {cereals,ground beef}                       => {spaghetti}         0.003066258 0.6764706  3.885303
# [10] {frozen vegetables,soup}                    => {milk}              0.003999467 0.5000000  3.858539

# Few products shows high confidence because these products are most frequently bought products and its not because of some association rule

rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.2))
#Visualizing the results
inspect(sort(rules, by="lift")[1:10])

#products which are bought 4 times a day

rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2, target="frequent itemsets"))
inspect(sort(rules[1:10]))


