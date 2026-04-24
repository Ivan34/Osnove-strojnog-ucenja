import numpy as np
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import sys
import os

# Učitavanje modela 
model = load_model('mnist_model.keras')
print("Model učitan.\n")

# Putanja do slike
# Možete proslijediti ime datoteke kao argument: python zadatak_3.py moja_slika.png
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = 'test.png'

if not os.path.exists(img_path):
    print(f"GREŠKA: Datoteka '{img_path}' nije pronađena.")
    print("Nacrtajte znamenku u Paint-u i pohranite je kao test.png u isti direktorij.")
    sys.exit(1)

# Učitavanje i priprema slike
# keras.utils.load_img čita sliku i mijenja joj veličinu na 28×28
# color_mode='grayscale' osigurava jednokanalnu (sivu) sliku
img = keras.utils.load_img(
    img_path,
    color_mode='grayscale',   # pretvori u sive tonove
    target_size=(28, 28)      # promijeni veličinu na 28×28 (kao MNIST)
)

img_array = keras.utils.img_to_array(img)   # PIL Image - numpy (28, 28, 1)

# MNIST ima BIJELE znamenke na CRNOJ pozadini (inverzno od tipičnih slika!)
# Ako crtamo crnom bojom na bijeloj podlozi (Paint), trebamo invertirati sliku.
img_array = 255 - img_array    # invertiranje: bijela pozadina - crna, crna znamenka - bijela

# Skaliranje na [0, 1]
img_array = img_array.astype("float32") / 255

# Dodavanje dimenzije batch-a: (28, 28, 1) - (1, 28, 28, 1)
# Mreža uvijek prima "batch" primjera, pa čak i za jedan primjer treba ovu dimenziju
img_batch = np.expand_dims(img_array, axis=0)

#Prikaz obrađene slike
plt.figure(figsize=(4, 4))
plt.imshow(img_array.squeeze(), cmap='gray')
plt.title('Slika kako ju mreža vidi (28×28, sivi tonovi)')
plt.axis('off')
plt.tight_layout()
plt.show()

# Klasifikacija 
y_pred_prob = model.predict(img_batch)          # oblik: (1, 10) – vjerojatnosti
y_pred      = np.argmax(y_pred_prob, axis=1)   # klasa s najvišom vjerojatnošću

print("=" * 40)
print(f"  Predviđena znamenka: {y_pred[0]}")
print(f"  Vjerojatnost:        {y_pred_prob[0, y_pred[0]]*100:.1f} %")
print("=" * 40)

# Ispis vjerojatnosti za sve klase
print("\nVjerojatnosti za sve klase:")
for klasa, prob in enumerate(y_pred_prob[0]):
    bar   = '█' * int(prob * 40)
    oznaka = ' ← PREDVIĐENO' if klasa == y_pred[0] else ''
    print(f"  {klasa}: {bar:<40}  {prob*100:5.1f} %{oznaka}")

#  Komentar o rezultatima

# Očekivani rezultati i komentar:

# Dobro nacrtane znamenke (debele linije, centrirane) - visoka točnost (>95 %)
# Tanke linije, rubovi slike - mreža može griješiti jer nije trenirana na takvim primjerima
# Kosa pisma ili nestandardni oblici - mreža može griješiti
# Slika nije invertirana - sve predviđa kao 0 ili 1 (mreža "vidi" prazan papir)

# MNIST mreža je trenirana isključivo na normaliziranim 28×28 sivim slikama
# pa je osjetljiva na:
#   - veličinu i poziciju znamenke unutar slike
#   - debljinu crte
#   - stil pisanja

# CNN (konvolucijska mreža) bila bi robustnija za ove varijacije.
