from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn . linear_model as lm

df = pd.read_csv("data_C02_emission.csv")


X = df[[
    "Engine Size (L)",
    "Cylinders",
    "Fuel Consumption City (L/100km)",
    "Fuel Consumption Hwy (L/100km)",
    "Fuel Consumption Comb (L/100km)",
    "Fuel Consumption Comb (mpg)"
]]
y = df[["CO2 Emissions (g/km)"]]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

feature = "Fuel Consumption City (L/100km)"

plt.scatter(X_train[feature], y_train, color="blue",alpha=0.5)
plt.scatter(X_test[feature], y_test, color="red",alpha=0.5)

plt.xlabel(feature)
plt.ylabel("CO2 Emissions")
plt.title("CO2 vs " + feature)


sc = MinMaxScaler ()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

plt.figure()
plt.hist(X_train.iloc[:, 2],  color="blue", alpha=0.5)
plt.hist(X_train_n[:, 2], color="red", alpha=0.5)
plt.show()

linearModel=lm.LinearRegression()
linearModel.fit(X_train_n ,y_train)