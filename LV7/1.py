import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)

    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                          centers=4,
                          cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                          random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)

    # 2 grupe
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)

    else:
        X = []

    return X


# Broj grupa za svaki nacin generiranja
optimal_K = {1: 3, 2: 3, 3: 4, 4: 2, 5: 2}

fig, axes = plt.subplots(5, 2, figsize=(12, 12))

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

for flagc in range(1, 6):
    X = generate_data(500, flagc)
    K = optimal_K[flagc]

    # Lijeva kolona: originalni podaci
    ax_left = axes[flagc - 1][0]
    ax_left.scatter(X[:, 0], X[:, 1], s=10, color='steelblue', alpha=0.6)
    ax_left.set_title(f'Način {flagc} – originalni podaci (K={K} grupe/a)')
    ax_left.set_xlabel('$x_1$')
    ax_left.set_ylabel('$x_2$')

    # Primjena K-means
    km = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=42)
    km.fit(X)
    labels = km.predict(X)
    centres = km.cluster_centers_

    # Desna kolona: obojeni klasteri
    ax_right = axes[flagc - 1][1]
    for k in range(K):
        mask = labels == k
        ax_right.scatter(X[mask, 0], X[mask, 1], s=10,
                         color=colors[k % len(colors)], alpha=0.6,
                         label=f'Grupa {k+1}')
    ax_right.scatter(centres[:, 0], centres[:, 1],
                     s=150, c='black', marker='X', zorder=5, label='Centri')
    ax_right.set_title(f'Način {flagc} – K-means rezultat (K={K})')
    ax_right.set_xlabel('$x_1$')
    ax_right.set_ylabel('$x_2$')
    ax_right.legend(fontsize=7, loc='best')

plt.tight_layout()
plt.show()

flagc = [1,2,3,4,5]
k_values = [2, 3, 4, 6, 10] # 5 različitih vrijednosti K

for flag in flagc:
    X = generate_data(500, flag)

    # Kreiramo pod-grafove (subplots) za svaku vrijednost K unutar istog flaga
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f'Eksperiment s različitim K za FLAG = {flag}', fontsize=16)

    for i, k in enumerate(k_values):
        # Inicijalizacija i treniranje K-means
        km = KMeans(n_clusters=k, init="random", n_init=5, random_state=0)
        labels = km.fit_predict(X)
        centroids = km.cluster_centers_

        # Crtanje rezultata na odgovarajući subplot
        axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)
        axes[i].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100)
        axes[i].set_title(f'K = {k}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#Komentar
# Način 1: 3 jasno odvojene grupe – K-means radi odlično.
# Način 2: 3 izdužene/dijagonalne grupe – K-means radi dobro jer su grupe linearno odvojive.
# Način 3: 4 grupe različitih gustoća – K-means radi zadovoljavajuće, ali grupe s većim
#           std mogu biti djelomično ispremještane zbog kuglaste pretpostavke algoritma.
# Način 4: 2 koncentrična kružnice – K-means NIJE prikladan jer pretpostavlja konveksne
#           oblike; grupiranje po kružnicama ne može se postići centroidima.
# Način 5: 2 polumjeseca – K-means NIJE prikladan iz istog razloga kao i način 4;
#           oblik podataka nije konveksan pa centroidi ne mogu pravilno razdvojiti grupe.