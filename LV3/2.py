import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')


plt.figure(figsize=(8, 5))
data['CO2 Emissions (g/km)'].plot(kind='hist', bins=25, edgecolor='black')
plt.title('Histogram emisije CO2 plinova')
plt.xlabel('Emisija CO2 (g/km)')
plt.ylabel('Broj vozila')
plt.show()


plt.figure(figsize=(8, 5))
fuelType = data['Fuel Type'].unique()
colors = plt.cm.get_cmap('Set1', len(fuelType))

for i, fuel in enumerate(fuelType):
    subset = data[data['Fuel Type'] == fuel]
    plt.scatter(subset['Fuel Consumption City (L/100km)'], subset['CO2 Emissions (g/km)'], 
                color=colors(i), label=fuel, alpha=0.6)

plt.title('Odnos gradske potrošnje goriva i emisije CO2')
plt.xlabel('Gradska potrošnja (L/100km)')
plt.ylabel('Emisija CO2 (g/km)')
plt.show()


plt.figure(figsize=(8, 5))
data.boxplot(column='Fuel Consumption Hwy (L/100km)', by='Fuel Type', grid=False)
plt.title('Izvangradska potrošnja obzirom na tip goriva')
plt.xlabel('Tip goriva')
plt.ylabel('Izvangradska potrošnja (L/100km)')
plt.show()


plt.figure(figsize=(8, 5))
brojpogorivu = data.groupby('Fuel Type').size()
brojpogorivu.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Broj vozila po tipu goriva')
plt.xlabel('Tip goriva')
plt.ylabel('Broj vozila')
plt.xticks(rotation=0)
plt.show()


plt.figure(figsize=(8, 5))
prosjekco2cilindri = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
prosjekco2cilindri.plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Prosječna CO2 emisija s obzirom na broj cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Prosječna CO2 emisija (g/km)')
plt.xticks(rotation=0)
plt.show()