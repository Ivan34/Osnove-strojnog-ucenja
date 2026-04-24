import numpy as np
from tensorflow import keras
from keras.models import load_model
from matplotlib import pyplot as plt

#Učitavanje modela
# Učitavamo prethodno naučeni model pohranjen u mapi 'FCN/'
model = load_model('mnist_model.keras')
model.summary()

#Učitavanje i priprema MNIST podataka
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

# Iste transformacije kao u zadatku 1!
x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)   # → (10000, 28, 28, 1)

#Predikcija na testnom skupu
y_pred_prob = model.predict(x_test_s)          # vjerojatnosti za svaku klasu
y_pred      = np.argmax(y_pred_prob, axis=1)   # predviđena klasa (0–9)
y_true      = y_test                            # prave oznake (originalne, ne one-hot)

#Pronalazak loše klasificiranih uzoraka
#Boolean maska: True = pogrešno klasificiran
wrong_mask    = y_pred != y_true
wrong_indices = np.where(wrong_mask)[0]   # indeksi pogrešnih primjera

print(f"Ukupno pogrešno klasificiranih: {len(wrong_indices)} / {len(y_true)}")
print(f"Točnost: {(1 - len(wrong_indices)/len(y_true))*100:.2f} %")

#Prikaz 12 loše klasificiranih slika
n_prikaz = 12
fig, axes = plt.subplots(3, 4, figsize=(10, 8))

for ax, idx in zip(axes.flat, wrong_indices[:n_prikaz]):
    ax.imshow(x_test[idx], cmap='gray')
    stvarna    = y_true[idx]
    predvidena = y_pred[idx]
    ax.set_title(
        f'Prava: {stvarna}  |  Predviđena: {predvidena}',
        color='red',
        fontsize=9
    )
    ax.axis('off')

plt.suptitle('Loše klasificirane slike iz testnog skupa', fontsize=13)
plt.tight_layout()
plt.savefig('misclassified.png', dpi=100)
plt.show()

print("\nSlika pohranjena kao 'misclassified.png'")