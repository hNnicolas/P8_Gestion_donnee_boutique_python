# Projet P8 - Analyse des ventes et gestion des stocks ![Logo Bottleneck](https://github.com/hNnicolas/P8_Gestion_donnee_boutique_python/blob/main/logo.png)

## 🎯 Objectifs

Ce projet répond à une mission confiée par l’entreprise **Bottleneck**.  
Il s’agit de :

1. **Phase 1** : Agréger et nettoyer les données issues de plusieurs sources (ERP, site web, table de liaison).
2. **Phase 2** : Réaliser des analyses stratégiques pour le CODIR :
   - Chiffre d’affaires par produit et total.
   - Analyse du top 20/80.
   - Détection des valeurs aberrantes (prix, marges, etc.).
   - Analyse des stocks (rotation, mois de stock, ruptures).
   - Étude des corrélations entre variables clés (prix, marge, ventes, etc.).

Les résultats sont exportés sous forme de graphiques et d’indicateurs synthétiques.  
Une **présentation PowerPoint** est également fournie pour le **CODIR**.

---

## ⚙️ Prérequis

- **Python** 3.9 ou plus
  pip install -r requirements.txt

## ▶️ Lancer l'analyse

Se placer dans le dossier scripts/ :
cd scripts

Lancer le script Python :

```bash
python notebook.py
```

## 📊 Résultats générés

Les graphiques (CA, boxplot prix, marges, stocks, corrélations) sont exportés dans le dossier :
plots/

Une synthèse chiffrée s’affiche dans le terminal :

CA total

Nombre de produits représentant 80% du CA

Produits avec stocks critiques

Produits avec prix aberrants

Produits sans correspondance ERP

### 1. Top 50 produits par CA et CA cumulatif (règle 20/80)

![Top 50 CA](https://github.com/hNnicolas/P8_Gestion_donnee_boutique_python/blob/main/plots/top50_ca.png)
_Top 50 produits triés par chiffre d’affaires et courbe du CA cumulé (ligne rouge)._

_(Variante relative)_  
`![Top 50 CA](./plots/top50_ca.png)`

---

### 2. Boxplot des prix

![Boxplot des prix](https://github.com/hNnicolas/P8_Gestion_donnee_boutique_python/blob/main/plots/boxplot_prix.png)
_Repère les valeurs aberrantes de prix (outliers) — utile pour détecter des erreurs de saisie ou produits mal tarifés._

_(Variante relative)_  
`![Boxplot des prix](./plots/boxplot_prix.png)`

---

### 3. Distribution des taux de marge

![Distribution des marges](https://github.com/hNnicolas/P8_Gestion_donnee_boutique_python/blob/main/plots/hist_marge.png)
_Histogramme de la distribution des taux de marge ; permet d’identifier dispersion et cas extrêmes._

_(Variante relative)_  
`![Distribution des marges](./plots/hist_marge.png)`

---

### 4. Distribution des mois de stock

![Distribution mois de stock](https://github.com/hNnicolas/P8_Gestion_donnee_boutique_python/blob/main/plots/hist_stock.png)
_Histogramme du nombre de mois de stock par produit — identifie produits à rotation lente ou risque de rupture._

_(Variante relative)_  
`![Distribution mois de stock](./plots/hist_stock.png)`

---

### 5. Heatmap des corrélations

![Heatmap corrélations](https://github.com/hNnicolas/P8_Gestion_donnee_boutique_python/blob/main/plots/heatmap_corr.png)
_Carte de corrélations entre variables clés (prix, prix d’achat, stock, CA, marges) — utile pour prioriser les analyses._

_(Variante relative)_  
`![Heatmap corrélations](./plots/heatmap_corr.png)`
