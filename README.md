# 🫁 Détection de pneumonie sur radiographies thoraciques

Classification d'images de radiographies pulmonaires (**NORMAL** vs **PNEUMONIA**) par apprentissage profond, avec un **EfficientNetB0** pré-entraîné (transfer learning), des cartes de chaleur **Grad-CAM** pour l'interprétabilité et une démo interactive **Gradio**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omrani-wiem/pneumonie-radiographie_/blob/main/radioagraphie.ipynb)

---

## 📌 Aperçu

Tout le projet est dans un notebook (`radioagraphie.ipynb`) conçu pour Google Colab :

1. Téléchargement du dataset depuis Kaggle
2. Préparation des données (normalisation, augmentation)
3. Gestion du déséquilibre des classes
4. Entraînement en deux phases (tête de classification, puis fine-tuning)
5. Évaluation (précision, rappel, F1, courbe ROC, seuil optimal)
6. Interprétation avec Grad-CAM
7. Démo web avec Gradio

---

## 📊 Données

- **Source** : [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) sur Kaggle (`paultimothymooney/chest-xray-pneumonia`)
- **Taille** : environ 2,3 Go

| Jeu | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| Entraînement | 1 341 | 3 875 | 5 216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

Les classes sont déséquilibrées : il y a presque trois fois plus de cas de pneumonie que de cas normaux.

---

## 🧠 Méthodologie

### Prétraitement
- Images redimensionnées en **224 × 224**, normalisées entre 0 et 1
- **Augmentation** (entraînement uniquement) : retournement horizontal, rotation (±6 %), zoom (±8 %)

### Modèle
- **Backbone** : EfficientNetB0 pré-entraîné sur ImageNet
- **Tête** : `GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(1, sigmoid)`
- **Perte** : entropie croisée binaire, avec **poids de classes** (NORMAL ≈ 1,94, PNEUMONIA ≈ 0,67)
- **Métriques suivies** : accuracy, AUC, précision, rappel

### Entraînement en deux phases
1. **Tête seule** : backbone gelé, learning rate 1e-3, jusqu'à 15 époques
2. **Fine-tuning** : dégel des 30 % de couches finales du backbone, learning rate 1e-5, 5 époques

Callbacks : `ModelCheckpoint` (meilleur `val_auc`), `EarlyStopping` (patience 5), `ReduceLROnPlateau`.

### Interprétabilité
Des cartes **Grad-CAM** (couche `top_conv`) montrent les zones de l'image qui influencent la décision du modèle.

---

## 📉 Résultats (jeu de test, 624 images)

**Seuil standard (0,5)**

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| NORMAL | 0,829 | 0,620 | 0,709 |
| PNEUMONIA | 0,802 | 0,923 | 0,858 |

Accuracy globale : **80,9 %**.

**Seuil ajusté (0,719, maximisant le F1)**

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| NORMAL | 0,769 | 0,855 | 0,810 |
| PNEUMONIA | 0,907 | 0,846 | 0,875 |

Accuracy globale : **84,9 %**.

> ⚠️ Le seuil de 0,719 a été choisi sur le jeu de test lui-même. Ce résultat est donc optimiste. Pour une évaluation propre, il faut le choisir sur un jeu de validation séparé.

## 🛠️ Technologies

- **Python 3**, **TensorFlow / Keras** (EfficientNetB0)
- `numpy`, `matplotlib`, `opencv-python`, `scikit-learn`
- `gradio` pour la démo
- Kaggle CLI pour le téléchargement des données

---

## 🚀 Utilisation

### Sur Google Colab (recommandé)
1. Cliquez sur le badge **Open In Colab** en haut de ce README.
2. Activez un GPU (Exécution → Modifier le type d'exécution).
3. Créez un jeton API Kaggle (kaggle.com → Settings → API) et importez le fichier `kaggle.json` quand le notebook le demande.
4. Exécutez les cellules dans l'ordre.

> 🔒 **Ne publiez jamais votre `kaggle.json`** ni la sortie de la cellule qui l'affiche. Videz les sorties du notebook (Edit → Clear all outputs) avant de le partager.

### En local

```bash
git clone https://github.com/omrani-wiem/pneumonie-radiographie_.git
cd pneumonie-radiographie_

pip install tensorflow numpy matplotlib scikit-learn opencv-python gradio kaggle jupyter
jupyter notebook radioagraphie.ipynb
```

---

## 📁 Structure du dépôt

```
pneumonie-radiographie_/
├── radioagraphie.ipynb   # Notebook principal
└── README.md
```

---

## 👤 Auteur

**Wiem Omrani** — [@omrani-wiem](https://github.com/omrani-wiem)

---

## ⚖️ Avertissement

Ce projet est réalisé à des fins **éducatives**. Il ne remplace en aucun cas un avis ou un diagnostic médical.
