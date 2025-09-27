# Projet P8 - Analyse des ventes et gestion des stocks (Bottleneck)

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
- Bibliothèques nécessaires :  
  ```bash
  pip install pandas numpy matplotlib seaborn scipy

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
