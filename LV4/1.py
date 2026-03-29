import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, 
                             mean_absolute_error, 
                             r2_score, 
                             mean_absolute_percentage_error)

data = pd.read_csv('data_C02_emission.csv')

features_num = [
    'Engine Size (L)', 
    'Cylinders', 
    'Fuel Consumption City (L/100km)', 
    'Fuel Consumption Hwy (L/100km)', 
    'Fuel Consumption Comb (L/100km)', 
    'Fuel Consumption Comb (mpg)'
]
X = data[features_num]
y = data['CO2 Emissions (g/km)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

for feature in features_num:
    plt.figure(figsize=(6, 4))
    plt.scatter(X_train[feature], y_train, color='blue', label='Skup za učenje', alpha=0.5)
    plt.scatter(X_test[feature], y_test, color='red', label='Skup za testiranje', alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Emisija CO2 (g/km)')
    plt.legend()
    plt.title(f'Ovisnost emisije CO2 o: {feature}')
    plt.show()


feature_to_plot = 'Fuel Consumption Comb (L/100km)'
feature_idx = features_num.index(feature_to_plot)

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.hist(X_train[feature_to_plot], bins=20, color='blue', alpha=0.7)
plt.title(f'{feature_to_plot}\nPrije skaliranja')


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)       


plt.subplot(1, 2, 2)
plt.hist(X_train_scaled[:, feature_idx], bins=20, color='green', alpha=0.7)
plt.title(f'{feature_to_plot}\nNakon skaliranja')
plt.show()


model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\n--- PARAMETRI MODELA ---")
print(f"Slobodni član (theta_0): {model.intercept_:.4f}")
for i, feature in enumerate(features_num):
    print(f"Koeficijent theta_{i+1} za '{feature}': {model.coef_[i]:.4f}")


y_pred = model.predict(X_test_scaled)

plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, color='purple', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Idealna procjena')
plt.xlabel('Stvarna emisija CO2 (g/km)')
plt.ylabel('Procijenjena emisija CO2 (g/km)')
plt.title('Stvarne vs. Procijenjene vrijednosti emisije CO2')
plt.legend()
plt.show()


mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse) 
mae = mean_absolute_error(y_test, y_pred) 
mape = mean_absolute_percentage_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred) 

print("\n--- VREDNOVANJE MODELA NA TESTNOM SKUPU ---")
print(f"Srednja kvadratna pogreška (MSE): {mse:.2f}")
print(f"Korijen iz srednje kvadratne pogreške (RMSE): {rmse:.2f}")
print(f"Srednja apsolutna pogreška (MAE): {mae:.2f}")
print(f"Srednja apsolutna postotna pogreška (MAPE): {mape:.4f}")
print(f"Koeficijent determinacije (R2): {r2:.4f}")