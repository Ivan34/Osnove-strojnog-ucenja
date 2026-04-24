import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()


#1. Broj različitih boja
w, h, d = img.shape
img_float = img.astype(np.float64) / 255.0
img_array = img.reshape(w * h, d)

unique_colors = np.unique(img.reshape(-1, 3), axis=0)
print(f"Dimenzije slike: {w} x {h} piksela, {d} kanala")
print(f"Ukupan broj piksela: {w * h}")
print(f"Broj različitih boja u originalnoj slici: {len(unique_colors)}")

#2 & 3. K-means kvantizacija za više vrijednosti K
K_values = [2, 4, 8, 16, 32]
inertias = []

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Originalna slika
axes[0][0].imshow(img)
axes[0][0].set_title(f'Originalna ({len(unique_colors)} boja)')
axes[0][0].axis('off')

for idx, K in enumerate(K_values):
    km = KMeans(n_clusters=K, init='k-means++', n_init=3,
                max_iter=200, random_state=42)
    km.fit(img_array)
    labels = km.labels_
    inertias.append(km.inertia_)

    # Zamijeni svaki piksel s njemu najbližim centrom
    img_approx = km.cluster_centers_[labels]
    img_quantized = img_approx.reshape(w, h, d)
    img_quantized = np.clip(img_quantized, 0, 1)

    row, col = divmod(idx + 1, 3)
    ax = axes[row][col]
    ax.imshow(img_quantized)
    ax.set_title(f'K = {K} boja')
    ax.axis('off')

plt.tight_layout()
plt.show()

#6. Lakat metoda – J vs K
K_elbow = list(range(1, 13))
inertias_elbow = []
for K in K_elbow:
    km = KMeans(n_clusters=K, init='k-means++', n_init=3,
                max_iter=150, random_state=42)
    km.fit(img_array)
    inertias_elbow.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_elbow, inertias_elbow, 'bo-', linewidth=2, markersize=6)
plt.xlabel('Broj grupa K')
plt.ylabel('J (inercija)')
plt.title('Lakat metoda – ovisnost J o broju grupa K')
plt.xticks(K_elbow)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#7. Binarne maske po grupama (za K=4)
K_mask = 4
km_mask = KMeans(n_clusters=K_mask, init='k-means++', n_init=5,
                 max_iter=200, random_state=42)
km_mask.fit(img_array)
labels_mask = km_mask.labels_.reshape(w, h)

fig, axes2 = plt.subplots(1, K_mask + 1, figsize=(4 * (K_mask + 1), 4))
axes2[0].imshow(img)
axes2[0].set_title('Originalna')
axes2[0].axis('off')

for k in range(K_mask):
    binary_mask = (labels_mask == k).astype(np.uint8) * 255
    axes2[k + 1].imshow(binary_mask, cmap='gray')
    c = (km_mask.cluster_centers_[k] * 255).astype(int)
    axes2[k + 1].set_title(f'Grupa {k+1}\nRGB≈({c[0]},{c[1]},{c[2]})')
    axes2[k + 1].axis('off')

fig.suptitle(f'Binarne maske po grupama (K={K_mask})', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()


#Komentari
# K=2-4:  Slika izgubi mnogo detalja, vide se samo dominantne boje.
# K=8-16: Dobar kompromis – slika vizualno slična originalu uz drastično manje boja.
# K=32+:  Gotovo identično originalu; dobitak je minimalan, a trošak računanja raste.
# Lakat:  Nagli pad J se uočava za mali K; točan "lakat" ovisi o sadržaju slike.
# Maske:  Svaka binarna slika odgovara jednoj grupi piksela (npr. pozadina, objekt,
#         sjena…). To je korisno za segmentaciju slike bez oznaka.