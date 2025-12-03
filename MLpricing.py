import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("housing.csv")

#preprocessing 
data.dropna(inplace=True)
data.drop(['ocean_proximity'],axis=1,inplace=True)#delete this later; needed it to view categorical data
#training/testing splitting
from sklearn.model_selection import train_test_split

x = data.drop(['median_house_value'],axis=1)
y = data['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #test size specifies how much of the data you want to test your model out on... typically its 80:20 split train:test



training_data = x_train.join(y_train)

plt.figure(figsize=(20,8))

sns.heatmap(training_data.corr(),annot=True,cmap="YlGnBu") #annot adds correlation labels to chart

plt.show()