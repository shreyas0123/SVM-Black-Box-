################################### problem1 ###############################
import pandas as pd
import numpy as np
from sklearn import preprocessing

salary_train = pd.read_csv("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Black  Box Tech (Support Vector Machine)/SalaryData_Train.csv")
salary_test = pd.read_csv("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Black  Box Tech (Support Vector Machine)/SalaryData_Test.csv")
salary_train.describe()
salary_test.describe()
salary_test.columns
salary_test.dtypes
#EDA
#Selecting required column
salary_train = salary_train.drop(salary_train.columns[[4,6,7,8,12]], axis = 1)
salary_test = salary_test.drop(salary_test.columns[[4,6,7,8,12]], axis = 1)

#Label Encoding for train and test data
label_encoder = preprocessing.LabelEncoder()
salary_train['workclass'] = label_encoder.fit_transform(salary_train['workclass'])
salary_train['education'] = label_encoder.fit_transform(salary_train['education'])
salary_train['occupation'] = label_encoder.fit_transform(salary_train['occupation'])
salary_train['Salary'] = label_encoder.fit_transform(salary_train['Salary'])

label_encoder = preprocessing.LabelEncoder()
salary_test['workclass'] = label_encoder.fit_transform(salary_test['workclass'])
salary_test['education'] = label_encoder.fit_transform(salary_test['education'])
salary_test['occupation'] = label_encoder.fit_transform(salary_test['occupation'])
salary_test['Salary'] = label_encoder.fit_transform(salary_test['Salary'])

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train_X = salary_train.iloc[:, 0:8]
train_y = salary_train.iloc[:, 8]
test_X  = salary_test.iloc[:, 0:8]
test_y  = salary_test.iloc[:, 8]


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear == test_y)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)


################################## problem2 ######################################
import pandas as pd
import numpy as np
from sklearn import preprocessing

forestfires_data = pd.read_csv("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Black  Box Tech (Support Vector Machine)/forestfires.csv")
forestfires_data.describe()
forestfires_data.columns

#EDA
#Label Encoding
label_encoder = preprocessing.LabelEncoder()
forestfires_data['month'] = label_encoder.fit_transform(forestfires_data['month'])
forestfires_data['day'] = label_encoder.fit_transform(forestfires_data['day'])
forestfires_data['size_category'] = label_encoder.fit_transform(forestfires_data['size_category'])

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train,test = train_test_split(forestfires_data, test_size = 0.20)

train_X = train.iloc[:, 0:30]
train_y = train.iloc[:, 30]
test_X  = test.iloc[:, 0:30]
test_y  = test.iloc[:, 30]


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear == test_y)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)

