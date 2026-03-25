# 🚀 Accélération de Scikit-Learn avec GPU (RAPIDS cuML)

Ce dépôt contient une série d'exemples pratiques montrant comment accélérer les algorithmes d'apprentissage automatique classiques de **Scikit-Learn** en utilisant la puissance des processeurs graphiques (GPU) via la bibliothèque **NVIDIA RAPIDS cuML**.

##  Description

L'objectif de ce projet est de démontrer l'utilisation de l'extension `%load_ext cuml.accel`. Cette fonctionnalité permet d'exécuter du code Scikit-Learn standard sur GPU avec des modifications minimales, offrant des gains de performance significatifs sur de grands ensembles de données.

### Algorithmes inclus :
- **Classification** : Random Forest, K-Nearest Neighbors (KNN), Régression Logistique.
- **Clustering** : K-Means, HDBSCAN.
- **Réduction de dimensionnalité** : PCA, UMAP.

## Prérequis

Pour exécuter ces notebooks, vous avez besoin de :
- Un environnement compatible avec **CUDA 12.x**.
- Un GPU NVIDIA (disponible gratuitement via Google Colab).
- Python 3.9+.

##  Installation

Avant d'exécuter les scripts, installez les dépendances nécessaires :

```bash
pip install cuml-cu12
pip install hdbscan umap-learn matplotlib
```

## Utilisation

L'activation de l'accélération GPU se fait simplement en ajoutant la ligne suivante au début de votre script ou notebook :

```python
%load_ext cuml.accel
import sklearn
# Vos imports sklearn habituels ici
```

## 📊 Résultats attendus

L'utilisation de `cuml.accel` permet de traiter des millions de lignes en quelques secondes, là où le CPU mettrait plusieurs minutes. Les résultats (précision, clusters, etc.) restent identiques à l'implémentation CPU standard.

## Ressources

- [Documentation NVIDIA RAPIDS](https://docs.rapids.ai/)
- [Dépôt GitHub cuML](https://github.com/rapidsai/cuml)

---
*Projet créé dans le cadre de l'exploration des performances GPU pour la Science des Données.*
