#-------------------------------------Sales Prediction Model----------------------------#


#----------------------------------Business Understanding-----------------------------#

#------------------------------------Defining the problem--------------------------------------#
#                                                                                              #
# Claire is the owner of a Grocery Store and has been facing issues with meeting the demand    #
# and thus incurring huge financial loss along with losing prospective customers. She          #
# needs a solution to help her better predict the future demand and meet the demand with       #
# optimal supply.                                                                              #
#                                                                                              #
#----------------------------------------------------------------------------------------------#



#-----------------------------------Why does it need to be solved------------------------------#
#                                                                                              #
# A solution will help Claire better prepare her inventory for future demands and              #
# thus not only help customers get what they want but would also help Claire to                #
# grow her business and earn more profits along with providing a better customer               #
# experience for consumers.                                                                    #
#----------------------------------------------------------------------------------------------#



#-----------------------------How would we try to solve the problem?---------------------------#
#                                                                                              #
# We will be using the power of machine learning to build linear predictive model              #
# using regression algorithms to predict the future sales for Claire's store and help          #
# her optimally strategise the future demand. It will also help her better understand          # 
# factors that affect her sales and thus make better business decisions.                       #
#----------------------------------------------------------------------------------------------#



#-----------------------------Data Understanding--------------------------#

#-------Set the working directory------------
getwd()                                                                         # Checking for the working directory
setwd("C:/Users/shris/OneDrive/Documents/Datasets")                             # Setting the working directory          


#-----------------Install the necessary packages-------------
install.packages("corrplot")                                                    # Installing the corrplot package for correlation matrix        
install.packages("car") 
install.packages("Information")
install.packages('healthcareai')  
install.packages("xlsx")


#--------  -Load the necessary packages---------------
library(corrplot)                                                               # graph of correlation matrix
library(car)                                                                    # Companion to Applied Regression
library(healthcareai)                                                           # build ML models with minimum code
library(tidyverse)                                                              # package for preparing, wrangling and visualizing data
library(caTools)                                                                # basic utility functions
library(xlsx)                                                                   # load data in spreadsheet format
library(MASS)                                                                   # Contains functions and datasets 
library(skimr)                                                                  # SUmmary Statistics
library(ggplot2)                                                                # Data Visualization
library(ggthemes)                                                               # Themes and scales
library(mlbench)                                                                # 
library(caret) 
library(Hmisc)
library(mlbench)
library(modelr)

#---------------Loading the dataset-------------
Individual <- read.xlsx("Individual.xlsx", sheetIndex = 1)                      # Loading the dataset into R console       



#-------------------Summarizing the dataset---------------

head(Individual, 5)                                                             # Check the top 5 observations of the data
str(Individual)                                                                 # Structure of the data

#-----------------------------------------------------------------------#
# 1) The Individual dataset has 150 observations with 7 features.       #
#                                                                       #
# 2) Almost all the features are in the numerical form but some of them #
#    seem to be categorical in nature.                                  #
#                                                                       #
# 3) Let's use the skim() function to better understand the structure   #
#    of the dataset                                                     #
#-----------------------------------------------------------------------#

summary(Individual)                                                             # Summary of the dataset

skim(Individual)

#------------------------------------------------------------------------------#
# 1) The dataset has no missing values.                                        #
# 2) Sometimes the missing values are present in the form of NA values, so     #
#    let's exdplictly check for NA values as well.                             #
#
#
#
#
#
#

sum(is.na(Individual))                                                          # 0 Na values present

# Check for class.
describe(Individual$Family.Income)
describe(Individual$Family.Size)
describe(Individual$Number.of.Vehicles)
describe(Individual$Distance.to.Store..KM.)
describe(Individual$Number.of.items)


#------------------------------------------------------------------------------------------#
#  1) We can clearly observe that the features Family size and Number of vehicles depict   #
#     categorical properties and thus need to be transformed accordingly.                  #
#                                                                                          #
#                                                                                          #
#                                                                                          #
#------------------------------------------------------------------------------------------#




missingness(Individual) %>%                                                     # Plot to check for missing values
  plot()


# Calculate skewness
skew <- apply(Individual[,2:7], 2, skewness)
print(skew)

# Check for correlation
corrplot(cor(Individual[,2:7]), method = "color")
corrplot(cor(Individual[,2:7]), method = "number")
corrplot(cor(Individual[,2:7]), method = "pie")


Individual$Family.Size <- as.factor(Individual$Family.Size)
Individual$Number.of.Vehicles <- as.factor(Individual$Number.of.Vehicles)

#------------------------------------------------------------------------------#
# 1) We can already see a strong correlation between the target variable and   #
#    family income, family size and number of items. The reason for the        #
#    correlation could be easily understood with the fact that higher family   #
#    income or more number of people in the family or buying more number of    #
#    items would be directly correlated with the sales transaction amount.     #
#------------------------------------------------------------------------------#





#-----------------------------Univariate analysis & Data Preparation------------

theme_set(theme_bw())


# Let's first analyze the decision variable 

ggplot(Individual, aes(x = X.Sales.Transaction)) +
  geom_histogram(binwidth = 20, col = "firebrick") +
  labs(title = "Distribution of Sales Transaction",
       subtitle = "Histogram",
       x = "Sales Transaction")

#------------------------------------------------------------------------------#
# 1) The decision variable seems to be normally distributed.                   #
# 2) Let's draw a boxplot to better understand the distribution and detect     #
#    the presence of outliers.                                                 #
#------------------------------------------------------------------------------#

ggplot(Individual, aes(x = X.Sales.Transaction)) +
  geom_boxplot(width = 3) +
  coord_cartesian(ylim = c(0, 1000)) +
  coord_flip() +
  scale_x_continuous() +
  scale_y_continuous() +
  ggtitle(" Box Plot of Sales Transaction") +                                      
  labs(title = "Sales Transaction",
       subtitle = "Outlier detection",
       x = "Sales Transactions in $")

quantile(Individual$X.Sales.Transaction, probs = c(0,0.95))
quantile(Individual$X.Sales.Transaction ,seq(0,1,0.05)) 
#-------------------------------------------------------------------------------#
# 1) The boxplot does show the presence of some outliers.                       #
# 2) The quantile function thus further confirm that any value over 389.12      #
#    could be considered as an outlier and may need treatment but however       #
#    our dataset has very limited observations and thus eliminating any         #
#    observation could have had detrimental affects on our model and we can miss#
#    some important information.                                                #
#-------------------------------------------------------------------------------#


# Boxplot distribution of Family Income

ggplot(Individual, aes(x = Family.Income)) +
  geom_boxplot(width = 3, col = "blue", fill = "firebrick") +
  coord_cartesian(ylim = c(0, 250000)) +
  coord_flip() +
  scale_x_continuous() +
  scale_y_continuous() +
  ggtitle(" Box Plot of Family Income") +                                      
  labs(title = "Family income",
       subtitle = "From Individual dataset",
       x = "Family Income")


quantile(Individual$Family.Income,seq(0,1,0.05))                # We can clearly observe that the two datapoints above 250000 are outliers

#-------------------------------------------------------------------------------#
# 1) Boxplot and the quantile function clearly shows that the feature has 2     #
#    outlier values which seems to be very far away from the distribution but   #
#    we still need to dig a bit deeper to analyze these 2 data points in order  #
#    to eliminate them out as outliers.                                         #
#-------------------------------------------------------------------------------#

Individual[Individual$Family.Income > 200000,]

#----------------------------------------------------------------#
# 1) A deeper view into other features of those 2 observations   #
#    seems to behave normal and the high family income could be  #
#    attributed to the large number of family members present.   #
#    thus we will not eliminate these observations.              #                       #
#                                                                #
#----------------------------------------------------------------#

# Barchart of Family Size feature.

ggplot(Individual, aes(Family.Size)) +
  geom_bar(alpha = 0.9, fill = "steelblue", width = 0.7, col = "green", position = "dodge") +
  ggtitle("Bar plot of the family size") +
  labs(title="Number of people in the family", 
       subtitle="From Individual dataset", 
       x = "Number of family members") 

# Histogram of number of vehicles.

ggplot(Individual, aes(Number.of.Vehicles)) +
  geom_bar(alpha = 0.9, fill = "steelblue", width = 0.7, col = "green", position = "dodge") +
  ggtitle("Bar plot of number of vehicles owned") +
  labs(title="Number of vehicles", 
       subtitle="From Individual dataset", 
       x = "Number of vehicles") 

# Histogram of  Number of items bought.

ggplot(Individual, aes(x = Number.of.items)) +
  geom_histogram(binwidth = 1, col = "firebrick") +
  labs(title = "Distribution of number of items bought",
       subtitle = "From the Individual Dataset",
       x = "Number of items bought")

quantile(Individual$Number.of.items,seq(0,1,0.05))

# Histogram of Distance to store in Kms.

ggplot(Individual, aes(x = Distance.to.Store..KM.)) +
  geom_histogram(binwidth = 1, col = "firebrick") +
  labs(title = "Distribution of Distance to store in Kms",
       subtitle = "From the Individual Dataset",
       x = "Distance to store in KMS")

quantile(Individual$Distance.to.Store..KM.,seq(0,1,0.05))




#------------------------Multivariate Analysis-------------------------------------------

# Let's analyze the decision variable against each feature.

# Family Income
str(Individual)
ggplot(Individual, aes(x=Family.Income, y= X.Sales.Transaction)) + 
  geom_point() +  
  geom_smooth(method="lm", col="firebrick", se=FALSE) + 
  coord_cartesian(xlim=c(0, 300000), ylim=c(0, 600)) + 
  scale_x_continuous(n.breaks = 6) +
  scale_y_continuous(n.breaks = 6) +
  labs(title="Family Size vs Sales Transaction",
       subtitle = "From the Individual Dataset",
       x = "Family Income",
       y = "Sales Transaction")

ggplot(Individual, aes(x=Family.Income, y= X.Sales.Transaction)) + 
  geom_point() +  
  geom_smooth(method="lm", col="firebrick", se=FALSE) + 
  coord_cartesian(xlim=c(0, 300000), ylim=c(0, 600)) + 
  scale_x_continuous(n.breaks = 6) +
  scale_y_continuous(n.breaks = 6) +
  labs(title="Family Size vs Sales Transaction by Family Size",
       subtitle = "From the Individual Dataset",
       x = "Family Income",
       y = "Sales Transaction") +
  facet_grid(Individual$Family.Size)

ggplot(Individual, aes(x=Family.Income, y= X.Sales.Transaction)) + 
  geom_point() +  
  geom_smooth(method="lm", col="firebrick", se=FALSE) + 
  coord_cartesian(xlim=c(0, 300000), ylim=c(0, 600)) + 
  scale_x_continuous(n.breaks = 6) +
  scale_y_continuous(n.breaks = 6) +
  labs(title="Family Size vs Sales Transaction by number of vehicles",
       subtitle = "From the Individual Dataset",
       x = "Family Income",
       y = "Sales Transaction") +
  facet_grid(Individual$Number.of.Vehicles)


# Family Size

ggplot(Individual, aes(y = X.Sales.Transaction, x = Family.Size)) +
  geom_boxplot() +
  ggtitle("Bar plot of sales transactions by family size") +
  labs(title="Sales by number of people in a family", 
       subtitle="From Individual dataset", 
       x = "Number of family members") 


# Number of Vehicles

ggplot(Individual, aes(y = X.Sales.Transaction, x = Number.of.Vehicles)) +
  geom_boxplot() +
  ggtitle("Bar plot of sales transactions by number of vehicles ") +
  labs(title="Sales by number of vehicles owned by a family", 
       subtitle="From Individual dataset", 
       x = "Number of vehicles") 

# Distance to store in Kms.


ggplot(Individual, aes(y = X.Sales.Transaction, x = as.factor(Distance.to.Store..KM.))) +
  geom_boxplot() +
  ggtitle("Box plot of Sales Transaction categorized by the distance to the store in kms") +
  labs(title="Sales by distance to the store", 
       subtitle="From Individual dataset", 
       x = "Distance to the store in kilometers") 
 


# Number of items bought

ggplot(Individual, aes(x=Number.of.items, y= X.Sales.Transaction)) + 
  geom_point() +  
  geom_smooth(method="lm", col="firebrick", se=FALSE) + 
  labs(title="Number of items bought vs Sales Transaction",
       subtitle = "From the Individual Dataset",
       x = "Number of items bought",
       y = "Sales Transaction")



#--------------------Spot Checking various ML algorithm using the Healthcare.AI package-------------------


str(Individual)
split_data <- split_train_test(d = Individual,
                              outcome = X.Sales.Transaction,
                              p = .7,
                              seed = 13333)

auto_models <- machine_learn(split_data$train, Observations,  outcome = X.Sales.Transaction )       
                                                                               
auto_models


predictions <- predict(auto_models, newdata = split_data$test)
predictions
plot(predictions)                                                             




#--Lets split the dataset


prepped_training_data <- prep_data(split_data$train, outcome = X.Sales.Transaction,
                                   center = TRUE, scale = TRUE)

Model_1 <- tune_models(d = prepped_training_data,
            outcome = X.Sales.Transaction,
            tune_depth = 25,
            metric = "RMSE")

evaluate(Model_1, all_models = TRUE)

models["Random Forest"] %>%
  plot()

interpret(Model_1) %>%
  plot()
 
get_variable_importance(Model_1) %>%
  plot()
 
explore(Model_1) %>%
  plot()

predictions_1 <- predict(Model_1, split_data$test)
plot(predictions_1)


# -- Lets build a linear model on the prepped data

set.seed(777)
trainControl <- trainControl(method="cv", number=10)


# Linear Regression
fit.lm <- train(X.Sales.Transaction~., data=split_data$train, method="lm", metric= c("RMSE", "Accuracy"), preProc=("scale"), trControl=trainControl)
print(fit.lm)
summary(fit.lm)
print(fit.lm$finalModel)
fit.lm$results


predictions.lm <- predict(fit.lm, split_data$test )
plot(split_data$test$X.Sales.Transaction,predictions.lm )

Accuracy <- R2(predictions.lm, split_data$test$X.Sales.Transaction)*100
Accuracy


save_models(fit.lm, file = "Sales_prediction_model.RDS")
