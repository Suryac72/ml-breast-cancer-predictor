# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

#Importing Dataset

dataset = pd.read_csv(r'D:\MyJavaCodes\Flask\Breast_cancer_data.csv', error_bad_lines=False)
x_feature = dataset.iloc[:,:-1]
y_target = dataset.iloc[:,-1]

#Spliting Dataset into Training set and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_feature,y_target,test_size = 0.25,random_state = 0)

#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Training our Dataset into LogisticRegression Algorithm

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)


# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[11, 17, 130,1020,1]]))
