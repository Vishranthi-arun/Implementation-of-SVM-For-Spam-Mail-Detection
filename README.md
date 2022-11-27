# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Vishranthi A
RegisterNumber:  212221230124
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
result

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![1](https://user-images.githubusercontent.com/93427278/204126417-3c757f24-0327-41e0-a3d4-90dd1a4528e7.png)



![2](https://user-images.githubusercontent.com/93427278/204126433-66650b34-6924-4038-87ea-5481eedd4917.png)



![3](https://user-images.githubusercontent.com/93427278/204126437-5f159cb8-684e-416c-adc6-c648f17a0c3d.png)



![4](https://user-images.githubusercontent.com/93427278/204126446-9e129516-60d8-4c08-b307-d59f247e318e.png)



![5](https://user-images.githubusercontent.com/93427278/204126454-842e31ec-ede1-4040-8dac-693c1c5b9df9.png)



![6](https://user-images.githubusercontent.com/93427278/204126457-a9afae17-4936-4a88-af9d-ed0fcb8a8612.png)



![7](https://user-images.githubusercontent.com/93427278/204126460-d35603b6-2f6f-445c-8c31-1055bc3378d8.png)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
