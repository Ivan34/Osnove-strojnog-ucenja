import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,classification_report, ConfusionMatrixDisplay

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', marker='o', label='Train', edgecolor='k')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', label='Test')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Podatkovni skup')
plt.legend()
plt.show()


LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

theta_0 = LogRegression_model.intercept_[0]
theta_1 = LogRegression_model.coef_[0][0]
theta_2 = LogRegression_model.coef_[0][1]
print(f"Parametri modela: theta_0={theta_0}, theta_1={theta_1}, theta_2={theta_2}")

x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x1_lin = np.linspace(x1_min, x1_max, 100)
x2_lin = (-theta_0 - theta_1 * x1_lin) / theta_2

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', marker='o', edgecolor='k')
plt.plot(x1_lin, x2_lin, color='red', label='Granica odluke')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Train podaci i granica odluke')
plt.legend()
plt.show()


y_pred = LogRegression_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Matrica zabune:\n", cm)
print("Tocnost:", accuracy_score(y_test, y_pred))
print("Preciznost:", precision_score(y_test, y_pred))
print("Odziv:", recall_score(y_test, y_pred))

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


boje = ['green' if y_test[i] == y_pred[i] else 'black' for i in range(len(y_test))]

plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=boje, marker='s', edgecolor='k')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Rezultati na testnom skupu (Zeleno: Točno, Crno: Pogrešno)')
plt.show()


print("\nZADATAK F: Model s više ulaznih veličina")


input_variables_extended = ['bill_length_mm', 'flipper_length_mm', 'bill_depth_mm', 'body_mass_g']


X_ext = pd.DataFrame[input_variables_extended].to_numpy()


X_train_ext, X_test_ext, y_train, y_test = train_test_split(X_ext, y, test_size=0.2, random_state=123)


model_ext = LogisticRegression(max_iter=2000)
model_ext.fit(X_train_ext, y_train)


y_pred_ext = model_ext.predict(X_test_ext)

print("Nova matrica zabune:\n", confusion_matrix(y_test, y_pred_ext))
print("Nova točnost:", accuracy_score(y_test, y_pred_ext))
print("\nNovi classification report:\n", classification_report(y_test, y_pred_ext))