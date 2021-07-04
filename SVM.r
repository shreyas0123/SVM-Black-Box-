##################################### problem1 ##################################################
#####Support Vector Machines 
library(readr)
# Load the Dataset
Salary_train <- read.csv("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Black  Box Tech (Support Vector Machine)\\SalaryData_Train.csv", stringsAsFactors = TRUE)
Salary_test <- read.csv("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Black  Box Tech (Support Vector Machine)\\SalaryData_Test.csv",stringsAsFactors = TRUE)

summary(Salary_train)
summary(Salary_test)
str(Salary_test)
str(Salary_train)

#checking column names
colnames(Salary_train)
colnames(Salary_test)

# Training a model on the data ----
# Begin by training a simple linear SVM
install.packages("kernlab")
library(kernlab)

salary_classifier <- ksvm(Salary ~ ., data = Salary_train, kernel = "vanilladot")
?ksvm

## Evaluating model performance ----
# predictions on testing dataset
salary_predictions <- predict(salary_classifier, Salary_test)

table(salary_predictions, Salary_test$Salary)
agreement <- salary_predictions == Salary_test$Salary
table(agreement)
prop.table(table(agreement))

## Improving model performance ----
salary_classifier_rbf <- ksvm(Salary ~ ., data = Salary_train, kernel = "rbfdot")
salary_predictions_rbf <- predict(salary_classifier_rbf, Salary_test)
agreement_rbf <- salary_predictions_rbf == Salary_test$Salary
table(agreement_rbf)
prop.table(table(agreement_rbf))

#85% the people who are living in that area having good salaries.Built properties can be easily sell.

########################### problem2 #################################################################
#####Support Vector Machines 

# Load the Dataset
forest_fires_data <- read.csv("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Black  Box Tech (Support Vector Machine)\\forestfires.csv", stringsAsFactors = TRUE)

summary(forest_fires_data)
colnames(forest_fires_data)

# Partition Data into train and test data
forest_train <- forest_fires_data[1:414, ]
forest_test  <- forest_fires_data[415:517, ]

# Training a model on the data ----
# Begin by training a simple linear SVM
install.packages("kernlab")
library(kernlab)

forest_classifier <- ksvm(size_category ~ ., data = forest_train, kernel = "vanilladot")
?ksvm

## Evaluating model performance ----
# predictions on testing dataset
forest_predictions <- predict(forest_classifier, forest_test)

table(forest_predictions, forest_test$size_category)
agreement <- forest_predictions == forest_test$size_category
table(agreement)
prop.table(table(agreement))

## Improving model performance ----
#forest_classifier_rbf <- ksvm(size_category ~ ., data = forest_train, kernel = "rbfdot")
#forest_predictions_rbf <- predict(forest_classifier_rbf, forest_test)
#agreement_rbf <- forest_predictions_rbf == forest_test$size_category
#table(agreement_rbf)
#prop.table(table(agreement_rbf))

########################## END #############################################

