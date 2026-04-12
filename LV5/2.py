import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

labels = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
  
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])
    plt.legend()
    plt.show()


df = pd.read_csv("penguins.csv")


print(df.isnull().sum())


df = df.drop(columns=['sex'])


df.dropna(axis=0, inplace=True)


df['species'] = df['species'].str.strip()
df['species'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
df.dropna(subset=['species'], inplace=True) 
df['species'] = df['species'].astype(int)  


output_variable = ['species']


input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy().ravel() 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


train_classes, train_counts = np.unique(y_train, return_counts=True)
test_classes, test_counts = np.unique(y_test, return_counts=True)

x_pozicije = np.arange(len(train_classes))
bar_width = 0.35

plt.figure(figsize=(8,4))
plt.bar(x_pozicije - bar_width/2, train_counts, bar_width, label='Train')
plt.bar(x_pozicije + bar_width/2, test_counts, bar_width, label='Test')
plt.xticks(x_pozicije, [labels[c] for c in train_classes])
plt.ylabel('Broj primjera')
plt.title('Raspodjela klasa u skupovima')
plt.legend()
plt.show()


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


print("--- Parametri modela ---")
print("Intercept (theta_0):\n", model.intercept_)
print("Koeficijenti (theta_1, theta_2):\n", model.coef_)
print("Razlika: Ovdje imamo po 3 intercepta i 3 para koeficijenata jer imamo 3 klase (višeklasni problem).")


plot_decision_regions(X_train, y_train, model)


y_pred = model.predict(X_test)

print("\n--- Evaluacija na testnom skupu ---")
print("Matrica zabune:\n", confusion_matrix(y_test, y_pred))
print("Točnost:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))


print("\nZADATAK F: Model s više ulaznih veličina")

input_variables_extended = ['bill_length_mm', 'flipper_length_mm', 'bill_depth_mm', 'body_mass_g']

X_ext = df[input_variables_extended].to_numpy()


X_train_ext, X_test_ext, y_train, y_test = train_test_split(X_ext, y, test_size=0.2, random_state=123)


model_ext = LogisticRegression(max_iter=2000)
model_ext.fit(X_train_ext, y_train)


y_pred_ext = model_ext.predict(X_test_ext)

print("Nova matrica zabune:\n", confusion_matrix(y_test, y_pred_ext))
print("Nova točnost:", accuracy_score(y_test, y_pred_ext))
print("\nNovi classification report:\n", classification_report(y_test, y_pred_ext))