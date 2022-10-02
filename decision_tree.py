# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def transform_array(x):
    res = [0,0,0,0]
    res[0] = 1 if x[0] == 'Yes' else 0
    res[1] = 1 if x[1] == 'Single' else 0
    res[2] = float(x[2][:-1])
    res[3] = 1 if x[3] == 'Yes' else 0
    return res
        

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']

for num, ds in enumerate(dataSets):

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)
    print(data_training)
    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float. 
    data_training = data_training.tolist()
    data_training_transform = list(map(transform_array, data_training))
    X = np.array([i[:-1] for i in data_training_transform])
   
    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    Y = np.array([i[-1] for i in data_training_transform])
    print(Y)

    #loop your training and test tasks 10 times here
    ave_acc = 0
    for i in range (10):

       #fitting the decision tree to the data by using Gini index and no max_depth
       clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
       clf = clf.fit(X, Y)
       
       #plotting the decision tree
       tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
       plt.show()

       #read the test data and add this data to data_test NumPy
       df_test = pd.read_csv('cheat_test.csv', sep=',', header=0)
       data_test = np.array(df_test.values)[:,1:]
       data_test = data_test.tolist()
       data_test = np.array(list(map(transform_array, data_test)))
       
       acc=0
       for data in data_test:
           #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           predict = clf.predict([data[:-1]])[0]
           
           #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
           correctness = 1 if predict==data[-1] else 0 
           acc += correctness
       run_acc = acc/len(data_test)
       
       #find the average accuracy of this model during the 10 runs (training and test set)
       ave_acc += run_acc

    #print the accuracy of this model during the 10 runs (training and test set).
    ave_acc = ave_acc/10
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    print(f'final accuracy when training on cheat_training_{num+1}.csv: {ave_acc}')