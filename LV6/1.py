import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

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
                    label=cl)


data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()


X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)


sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))


LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)


y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))


plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()


print("\nZADATAK 6.5.1: KNN")

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_n, y_train)

y_train_p_knn = knn_model.predict(X_train_n)
y_test_p_knn = knn_model.predict(X_test_n)

print("KNN (K=5) Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_knn)))
print("KNN (K=5) Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_p_knn)))


plot_decision_regions(X_train_n, y_train, classifier=knn_model)
plt.title("KNN Granica odluke (K=5)")
plt.xlabel('Age (skalirano)')
plt.ylabel('Estimated Salary (skalirano)')
plt.show()

for k in [1, 100]:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_n, y_train)
    plot_decision_regions(X_train_n, y_train, classifier=knn_temp)
    plt.title(f"KNN Granica odluke (K={k})")
    plt.xlabel('Age (skalirano)')
    plt.ylabel('Estimated Salary (skalirano)')
    plt.show()
    

print("\nZADATAK 6.5.2: KNN Grid Search")
param_grid_knn = {'n_neighbors': np.arange(1, 51)}
knn_gscv = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
knn_gscv.fit(X_train_n, y_train)

print("Najbolji parametri za KNN:", knn_gscv.best_params_)
print("Najbolja točnost unakrsne validacije za KNN:", "{:0.3f}".format(knn_gscv.best_score_))


print("\nZADATAK 6.5.3: SVM")
svm_model = svm.SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train_n, y_train)

y_test_p_svm = svm_model.predict(X_test_n)
print("SVM (RBF, C=1, gamma='scale') Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_p_svm)))

plot_decision_regions(X_train_n, y_train, classifier=svm_model)
plt.title("SVM Granica odluke (RBF, C=1)")
plt.xlabel('Age (skalirano)')
plt.ylabel('Estimated Salary (skalirano)')
plt.show()

svm_linear = svm.SVC(kernel='linear', C=1)
svm_linear.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=svm_linear)
plt.title("SVM Granica odluke (Linear, C=1)")
plt.xlabel('Age (skalirano)')
plt.ylabel('Estimated Salary (skalirano)')
plt.show()


print("\nZADATAK 6.5.4: SVM Grid Search")
param_grid_svm = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01]}
svm_gscv = GridSearchCV(svm.SVC(kernel='rbf'), param_grid_svm, cv=5, scoring='accuracy')
svm_gscv.fit(X_train_n, y_train)

print("Najbolji parametri za SVM:", svm_gscv.best_params_)
print("Najbolja točnost unakrsne validacije za SVM:", "{:0.3f}".format(svm_gscv.best_score_))