import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


reg = LinearRegression()
data = pd.read_csv("housing.csv")

#       <--- data preprocessing --->

data.dropna(inplace=True)

x = data.drop(['median_house_value'],axis=1) #include the whole testing set without predicting column
y = data['median_house_value'] #includes only predicting column

#test size specifies how much of the data you want to test your model out on... typically its 80:20 split train:test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


training_data = x_train.join(y_train)

plt.figure(figsize=(20,8))


#      <--- Feature Engineering --->
training_data['ppl_Per_HH'] = training_data['population'] / training_data['households']
training_data['roomsPerHousehold'] = training_data['total_rooms'] / training_data['households']

#Because some data is skewed we can create a log function in order to scale out a normally distributed bell curve
training_data['total_rooms'] = np.log(training_data['total_rooms'] + 1)
training_data['total_bedrooms'] = np.log(training_data['total_bedrooms'] + 1)
training_data['population'] = np.log(training_data['population'] + 1)
training_data['households'] = np.log(training_data['households'] + 1)

training_data =  training_data.join(pd.get_dummies(training_data.ocean_proximity))



training_data = training_data.drop(['ocean_proximity'],axis = 1)
#sns.scatterplot(x="median_income",y = "longitude",data=training_data, hue = "median_house_value",palette= "Spectral")



#   <--- Data Visualization --->
#sns.heatmap(training_data.corr(),annot=True,cmap="YlGnBu")


x_train, y_train = training_data.drop(['median_house_value'],axis=1),training_data['median_house_value']

reg.fit(x_train,y_train)

# <--- test data --->

test_data = x_test.join(y_test)


#feature engineering
test_data['ppl_Per_HH'] = test_data['population'] / test_data['households']
test_data['roomsPerHousehold'] = test_data['total_rooms'] / test_data['households']

#unskewing heavy tailed data
test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)

#encoding cat variables and dropping cat *cannot use cat.codes for lin reg because machine will think 1<2<3<4
test_data =  test_data.join(pd.get_dummies(test_data.ocean_proximity))
test_data = test_data.drop(['ocean_proximity'],axis=1)


reg.score(x_test,y_test)