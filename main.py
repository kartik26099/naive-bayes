import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
df=pd.read_csv(r"D:\coding journey\aiml\python\task\data set of ML project\classification\Naive-Bayes-Classification-Data.csv")
x=df.iloc[:, :-1].values
y=df.iloc[:, -1].values
sns.pairplot(df,hue="diabetes")
plt.show()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)
print(accuracy_score(y_test,y_predict))
classifier2=SVC(kernel="rbf",random_state=0)
classifier2.fit(x_train,y_train)
y_predict2=classifier2.predict(x_test)
print(accuracy_score(y_test,y_predict2))