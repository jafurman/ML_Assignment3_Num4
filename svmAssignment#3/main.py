# -------------------------------------------------------------------------
# AUTHOR: Joshua Furman FILENAME:
# support_vector_machine.py SPECIFICATION: This program trains and tests a support vector machine classifier with
# different hyperparameters for question 4 of the homework FOR: CS 4210- Assignment #3 TIME SPENT: 2 Hours
# IMPORTANT NOTE: THIS CODE REQUIRES NUMPY AND PANDAS PYTHON LIBRARIES.
# import required libraries
# importing some Python libraries
# -------------------------------------------------------------------------

from sklearn import svm
import numpy as np
import pandas as pd

# defining the hyperparameter values
c_values = [1, 5, 10, 100]
degree_values = [1, 2, 3]
kernel_values = ["linear", "poly", "rbf"]
decision_function_shape_values = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None)  # reading the training data by using Pandas library

X_training = np.array(df.values)[:,
             :64]  # getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,
             -1]  # getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None)  # reading the training data by using Pandas library

X_test = np.array(df.values)[:,
         :64]  # getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,
         -1]  # getting the last field to create the class testing data and convert them to NumPy array

# created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
HYPERPARAMETER = []
MAX_ACCURACY = float('-inf')
for c in c_values:
    for degree in degree_values:
        for kernel in kernel_values:
            for decision_function_shape in decision_function_shape_values:

                # Create an SVM classifier that will test all combinations of c, degree, kernel,
                # and decision_function_shape.
                clf = svm.SVC(C=c, degree=degree, kernel=kernel, decision_function_shape=decision_function_shape)

                # Fit SVM to the training data
                clf.fit(X_training, y_training)

                # make the SVM prediction for each test sample and start computing its accuracy
                count = 0
                for (x_testSample, y_testSample) in zip(X_test, y_test):
                    if clf.predict([x_testSample])[0] == y_testSample:
                        count += 1
                ACCURACY = count / len(y_test)

                # check if the calculated accuracy is higher than the previously one calculated. If so, update the
                # highest accuracy and print it together
                if MAX_ACCURACY < ACCURACY:
                    MAX_ACCURACY = ACCURACY
                    print('Highest SVM accuracy so far: {0}, Parameters: c={1}, degree={2}, kernel={3}, '
                          'decision_function_shape={4}'.format(MAX_ACCURACY, c, degree, kernel,
                                                               decision_function_shape))
