import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Parametri modela
num_classes = 10
input_shape = (28, 28, 1)

#Učitavanje podataka
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test:  X=%s, y=%s' % (x_test.shape, y_test.shape))

#TODO 1: Prikaz nekoliko slika iz train skupa
# Prikazujemo prvih 9 slika s njihovim oznakama (labelama)
fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i], cmap='gray')          # prikaži sliku u sivim tonovima
    ax.set_title(f'Oznaka: {y_train[i]}')        # ispiši pravu oznaku
    ax.axis('off')
plt.suptitle('Primjeri iz train skupa', fontsize=13)
plt.tight_layout()
plt.savefig('train_samples.png', dpi=100)
plt.show()

# skaliranje i oblikovanje podataka 
# Slike se skaliraju s [0, 255] → [0, 1]  (lakše učenje, stabilan gradijent)
x_train_s = x_train.astype("float32") / 255
x_test_s  = x_test.astype("float32")  / 255

# Keras Conv2D / Dense zahtijeva oblik (n_uzoraka, visina, širina, kanali)
x_train_s = np.expand_dims(x_train_s, -1)   # → (60000, 28, 28, 1)
x_test_s  = np.expand_dims(x_test_s,  -1)   # → (10000, 28, 28, 1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0],  "test samples")

# 1-od-K (one-hot) kodiranje oznaka
# npr. oznaka 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s  = keras.utils.to_categorical(y_test,  num_classes)


#TODO 2: Kreiranje modela (Sequential API)
# Arhitektura prema slici 8.5 iz upute:
#   Ulaz:          784 elemenata (flatten 28×28)
#   Skriveni 1:    100 neurona, ReLU
#   Skriveni 2:     50 neurona, ReLU
#   Izlaz:          10 neurona, softmax

model = keras.Sequential(
    [
        layers.Input(shape=input_shape),          # ulazni sloj: (28, 28, 1)
        layers.Flatten(),                          # razvlači 2D sliku u 1D vektor (784,)
        layers.Dense(100, activation="relu"),      # 1. skriveni sloj
        layers.Dense(50,  activation="relu"),      # 2. skriveni sloj
        layers.Dense(num_classes, activation="softmax"),  # izlazni sloj
    ]
)

model.summary()
# Ispis prikazuje:
#   - naziv i tip svakog sloja
#   - dimenziju izlaza (None = batch_size, koji može biti bilo koji broj)
#   - broj parametara (težine + bias-i) u svakom sloju
#   Ukupno ~ 109 410 trenabilnih parametara


#TODO 3: Konfiguracija procesa učenja (.compile)
# loss:       categorical_crossentropy → standardni gubitak za višeklasnu klasifikaciju
#             s 1-od-K kodiranim oznakama
# optimizer:  adam → adaptivna stopa učenja, bolja konvergencija od obične SGD
# metrics:    accuracy → pratimo točnost na skupu za učenje i validacijskom skupu
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)


#TODO 4: Pokretanje učenja (.fit)
batch_size = 64    # koliko se primjera obrađuje u jednom koraku (minibatch)
epochs     = 15    # koliko puta prolazimo kroz cijeli skup za učenje

# validation_split=0.1 → 10 % train skupa (6 000 primjera) postaje validacijski skup;
# mreža NE uči na tim primjerima, ali ih koristimo za praćenje generalizacije
history = model.fit(
    x_train_s,
    y_train_s,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
)

# Prikaz krivulja učenja
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['loss'],     label='Gubitak (train)')
ax1.plot(history.history['val_loss'], label='Gubitak (val)')
ax1.set_xlabel('Epoha'); ax1.set_ylabel('Gubitak')
ax1.set_title('Krivulja gubitka'); ax1.legend()

ax2.plot(history.history['accuracy'],     label='Točnost (train)')
ax2.plot(history.history['val_accuracy'], label='Točnost (val)')
ax2.set_xlabel('Epoha'); ax2.set_ylabel('Točnost')
ax2.set_title('Krivulja točnosti'); ax2.legend()

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=100)
plt.show()


#TODO 5: Evaluacija na testnom skupu + matrica zabune
score = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"\nTest loss:     {score[0]:.4f}")
print(f"Test accuracy: {score[1]*100:.2f} %")
# Očekujemo ~97–98 % točnosti na testnom skupu

# Predikcija za testni skup
y_pred_prob = model.predict(x_test_s)          # vjerojatnosti (10 klasa)
y_pred      = np.argmax(y_pred_prob, axis=1)   # uzimamo klasu s max. vjerojatnošću
y_true      = np.argmax(y_test_s, axis=1)      # prave oznake (decode iz one-hot)

# Matrica zabune (Confusion Matrix)
# Redak i = prava klasa, stupac j = predviđena klasa
# Idealno: samo dijagonala ima vrijednosti ≠ 0
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))

fig, ax = plt.subplots(figsize=(8, 7))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Matrica zabune – testni skup')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
plt.show()

print("\nMatrica zabune:")
print(cm)


#TODO 6: Pohranjivanje modela
# Model se pohranjuje u Keras-ov SavedModel format (direktorij FCN/)
# Sadrži: arhitekturu, naučene težine i konfiguraciju kompajlera
model.save("mnist_model.keras")
print("\nModel uspješno pohranjen u mapu")