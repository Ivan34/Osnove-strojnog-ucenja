import numpy as np
import pandas as pd

data = pd . read_csv ("data_C02_emission.csv")

print("Number of messurments:", len(data))
print(data.dtypes)
print(data.isnull().sum())
print(data.duplicated().sum())

data.dropna(axis=0)
data.dropna(axis=1)
data.drop_duplicates()
data = data.reset_index(drop=True)

maxThree = data.nlargest(3, "Fuel Consumption City (L/100km)")
minThree = data.nsmallest(3, "Fuel Consumption City (L/100km)")

print("The 3 with the highest city consumption: \n", maxThree[["Make", "Model","Fuel Consumption City (L/100km)"]])
print("The 3 with the lowest city consumption: \n", minThree[["Make", "Model","Fuel Consumption City (L/100km)"]])

betweenData = (data[(data["Engine Size (L)"] >= 2.5) & (data["Engine Size (L)"] <= 3.5)])
print(len(betweenData))
print(betweenData[["CO2 Emissions (g/km)"]].mean())

audiVozila = data[data["Make"] == "Audi"]
print("Number of audi cars: ", len(audiVozila))

print("co2 of audi with 4 cylinders: ", len(audiVozila[audiVozila["Cylinders"] == 4]))

evenCylinders = data[(data["Cylinders"] >= 4) & (data["Cylinders"] %2 == 0)]
print(len(evenCylinders))

cylGroup = data.groupby("Cylinders")
print("CO2 by cylinder count: ", cylGroup["CO2 Emissions (g/km)"].mean())

dizel = data[data['Fuel Type'] == 'D']['Fuel Consumption City (L/100km)']
benzin = data[data['Fuel Type'] == 'X']['Fuel Consumption City (L/100km)']

print(f"\nDizel - Avrage: {dizel.mean():.2f}, Median: {dizel.median():.2f}")
print(f"Regularni benzin - Avrage: {benzin.mean():.2f}, Medijan: {benzin.median():.2f}")


dizel_4_cil = data[(data['Fuel Type'] == 'D') & (data['Cylinders'] == 4)]
max_dizel_4_cil = dizel_4_cil.nlargest(1, 'Fuel Consumption City (L/100km)')
print("\n4 cylinder vehicles with the biggest city consuption:")
print(max_dizel_4_cil[['Make', 'Model', 'Fuel Consumption City (L/100km)']])


manual = data[data['Transmission'].str.startswith('M')]
print(f"\nAmount of manual vehicles: {len(manual)}")


correlation = data.corr(numeric_only=True)
print("\nCorelation matrix:")
print(correlation)


