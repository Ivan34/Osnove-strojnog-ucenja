import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import os

# 1. Ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 2. Pripremi podatke (skaliraj ih na raspon [0,1])
X_train_n = X_train.astype('float32') / 255.0
X_test_n = X_test.astype('float32') / 255.0

# 3. 1-od-K kodiranje (Popravljeno: uklonjen dtype koji uzrokuje TypeError)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 4. Izgradnja CNN mreze (Zadatak 9.4.2 - Dodani Dropout slojevi)
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25), # Sprjecava overfitting
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(500, activation='relu'),
    layers.Dropout(0.5), # Jacu dropout na potpuno povezanom sloju
    layers.Dense(10, activation='softmax')
])

model.summary()

# 5. Rjesavanje greske s logovima na Windowsima (FailedPreconditionError)
log_dir = "C:/temp/logs/cnn_dropout"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 6. Callbacks (Zadatak 9.4.3 - EarlyStopping)
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=100),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

# 7. Kompilacija i ucenje
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_n,
          y_train,
          epochs=40,
          batch_size=64,
          callbacks=my_callbacks,
          validation_split=0.1)

# 8. Evaluacija modela na testnom skupu
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}%')




# ZADATAK 9.4.1: Analiza pocetne mreze
# - Sto se dogodilo tijekom ucenja? 
#   U originalnoj mrezi bez dropouta dolazi do izrazajnog overfittinga. Tocnost na 
#   trening skupu ide do 100%, dok na validacijskom skupu stagnira ili pada. 
#   Gubitak (loss) na validaciji pocinje rasti nakon nekoliko epoha.

# ZADATAK 9.4.2: Utjecaj dropout slojeva
# - Kako komentirate utjecaj dropout slojeva?
#   Dropout nasumicno iskljucuje neurone i time prisiljava mrezu da uci robusnije znacajke. 
#   To je smanjilo overfitting, krivulje treninga i validacije su blize, a tocnost 
#   na testnom skupu se poboljsala (postignuto 79.24%).

# ZADATAK 9.4.3: Early Stopping
# - Uloga Early Stoppinga:
#   Algoritam je prekinuo ucenje cim je primijetio da se gubitak na validacijskom 
#   skupu nije smanjio u zadnjih 5 epoha. To sprjecava nepotrebno trosenje vremena
#   i dodatno sprjecava overfitting.

# ZADATAK 9.4.4: Teorijska pitanja
# 1. Batch size: 
#    Jako mali batch size (npr. 1) cini ucenje vrlo nestabilnim i sporim.
#    Jako veliki batch size moze dovesti do toga da model zapne u lokalnom minimumu.
# 2. Learning rate:
#    Premali learning rate znaci presporo ucenje (trebale bi tisuce epoha).
#    Preveliki learning rate uzrokuje "skakanje" preko rjesenja, pa model nikad ne konvergira.
# 3. Smanjenje mreze:
#    Smanjenjem broja slojeva mreza gubi "kapacitet" za ucenje slozenih uzoraka, 
#    sto dovodi do slabije tocnosti (underfitting).
# 4. Smanjenje podataka (50%):
#    S manje podataka mreža ce prebrzo nauciti primjere napamet (overfitting)
#    i nece moci prepoznati nove slike izvan skupa za ucenje.
