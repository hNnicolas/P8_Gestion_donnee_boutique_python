#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notebook Python pour CODIR
Analyse ventes, gestion stock, export graphiques PNG pour PPTX
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_theme(style="darkgrid")

# -------------------------------
# CHEMINS ET CHARGEMENT DES DONNÉES
# -------------------------------
data_path = Path(__file__).parent.parent / "data"
plots_path = Path(__file__).parent.parent / "plots"
plots_path.mkdir(exist_ok=True)

erp = pd.read_csv(data_path / "erp.csv")
web = pd.read_csv(data_path / "web.csv")
liaison = pd.read_csv(data_path / "liaison.csv")

# -------------------------------
# PHASE 1 : NETTOYAGE ET JOINTURE
# -------------------------------
def norm(x):
    try:
        return str(x).strip().upper()
    except:
        return x

erp['ref_erp_norm'] = erp['product_id'].astype(str).apply(norm)
liaison['ref_erp_norm'] = liaison['product_id'].astype(str).apply(norm)
liaison['ref_web_norm'] = liaison['id_web'].astype(str).apply(norm)
web['ref_web_norm'] = web['sku'].astype(str).apply(norm)

full = web.merge(liaison[['ref_web_norm','ref_erp_norm']], how='left', on='ref_web_norm')\
           .merge(erp, how='left', on='ref_erp_norm', suffixes=('_web','_erp'))

full['price_num'] = pd.to_numeric(full['price'], errors='coerce')
full['stock_quantity_num'] = pd.to_numeric(full['stock_quantity'], errors='coerce').fillna(0).astype(int)
full['purchase_price_num'] = pd.to_numeric(full['purchase_price'], errors='coerce')
full['a_valider_liaison'] = full['ref_erp_norm'].isna()

df = full.copy()
qty_col = 'total_sales'
price_col = 'price_num'

# -------------------------------
# PHASE 2 : ANALYSES
# -------------------------------
# CA par produit et total
if qty_col in df.columns and price_col in df.columns:
    df['total_sales_value'] = df[qty_col] * df[price_col]
    ca_total = df['total_sales_value'].sum()
else:
    raise ValueError("Colonnes pour qty ou price manquantes. CA non calculable.")

# Top 20/80
df_sorted = df.sort_values(by='total_sales_value', ascending=False)
df_sorted['ca_cumsum'] = df_sorted['total_sales_value'].cumsum()
df_sorted['ca_cumperc'] = df_sorted['ca_cumsum'] / ca_total * 100
top_20_percent = df_sorted[df_sorted['ca_cumperc'] <= 80]

# Outliers prix
df['price_zscore'] = np.abs(stats.zscore(df[price_col].fillna(0)))
Q1 = df[price_col].quantile(0.25)
Q3 = df[price_col].quantile(0.75)
IQR = Q3 - Q1
df['price_outlier_iqr'] = ((df[price_col] < Q1 - 1.5*IQR) | (df[price_col] > Q3 + 1.5*IQR))
outliers_price = df[(df['price_zscore'] > 3) | (df['price_outlier_iqr'])]

# Marges et rotation stock
df['margin_rate'] = ((df[price_col] - df['purchase_price_num']) / df[price_col] * 100).round(2)
df['rotation_stock'] = df['total_sales_value'] / df['stock_quantity_num'].replace(0, np.nan)
df['mois_stock'] = df['stock_quantity_num'] / df['total_sales_value'].replace(0, np.nan)
risque_stock = df[df['mois_stock'] < 1]

# Tableaux pour slides
top10 = df_sorted[['ref_web_norm','total_sales_value']].head(10)
prix_aberrants = outliers_price[['ref_web_norm', price_col,'price_zscore']].head(10)
stock_critique = risque_stock[['ref_web_norm','stock_quantity_num','total_sales_value','mois_stock']].head(10)
web_sans_erp = df[df['a_valider_liaison']][['ref_web_norm', 'total_sales', 'price_num']]

# -------------------------------
# FONCTION D'EXPORT GRAPHIQUES
# -------------------------------
def save_plot_png(fig, filename, tight=True, dpi=150):
    fig.savefig(filename, bbox_inches='tight' if tight else None, dpi=dpi)
    plt.close(fig)
    return str(filename)

# -------------------------------
# EXPORT GRAPHIQUES PNG
# -------------------------------

# 1️⃣ Top 20/80 CA
top_n = 50  # nombre de produits affichés pour lisibilité
df_topn = df_sorted.head(top_n).copy()
df_topn['ca_cumsum'] = df_topn['total_sales_value'].cumsum()
df_topn['ca_cumperc'] = df_topn['ca_cumsum'] / ca_total * 100

# Création du graphique
fig, ax = plt.subplots(figsize=(14,7))
ax.bar(range(len(df_topn)), df_topn['total_sales_value'], color='skyblue', label='CA produit')
ax2 = ax.twinx()
ax2.plot(range(len(df_topn)), df_topn['ca_cumperc'], color='red', marker='o', label='CA cumulé (%)')
ax2.axhline(80, color='orange', linestyle='--', label='80% du CA')

# Labels et titres
ax.set_xlabel("Produits triés par CA décroissant", fontsize=12)
ax.set_ylabel("CA produit (€)", fontsize=12)
ax2.set_ylabel("CA cumulé (%)", fontsize=12)
ax.set_xticks(range(len(df_topn)))
ax.set_xticklabels(df_topn['ref_web_norm'], rotation=90, fontsize=8)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title("Top 50 produits et CA cumulatif", fontsize=16, fontweight='bold')
plt.tight_layout()

# -------------------------------
# Sauvegarde dans /plots
# -------------------------------
plots_path = Path(__file__).parent.parent / "plots"
plots_path.mkdir(exist_ok=True)
file_path = plots_path / "top50_ca.png"

fig.savefig(file_path, bbox_inches='tight', dpi=150)
plt.close(fig)

print(f"Graphique Top 50 CA sauvegardé : {file_path}")


# 2️⃣ Boxplot prix
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(x=df['price_num'], ax=ax)
ax.set_title("Boxplot des prix produits", fontsize=16, fontweight='bold')
ax.set_xlabel("Prix (€)", fontsize=14)
save_plot_png(fig, plots_path / "boxplot_prix.png")

# 3️⃣ Histogramme marges
fig, ax = plt.subplots(figsize=(10,5))
sns.histplot(df['margin_rate'].dropna(), bins=20, kde=True, ax=ax, color='skyblue')
ax.set_title("Distribution des taux de marge (%)", fontsize=16, fontweight='bold')
ax.set_xlabel("Taux de marge (%)", fontsize=14)
ax.set_ylabel("Nombre de produits", fontsize=14)
save_plot_png(fig, plots_path / "hist_marge.png")

# 4️⃣ Histogramme mois de stock
fig, ax = plt.subplots(figsize=(10,5))
sns.histplot(df['mois_stock'].dropna(), bins=20, kde=True, ax=ax, color='orange')
ax.set_title("Distribution des mois de stock par produit", fontsize=16, fontweight='bold')
ax.set_xlabel("Mois de stock", fontsize=14)
ax.set_ylabel("Nombre de produits", fontsize=14)
save_plot_png(fig, plots_path / "hist_stock.png")

# 5️⃣ Heatmap corrélations
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df[['price_num','purchase_price_num','stock_quantity_num','total_sales_value','margin_rate']].corr(),
            annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title("Corrélations entre variables clés", fontsize=16, fontweight='bold')
save_plot_png(fig, plots_path / "heatmap_corr.png")

# -------------------------------
# SYNTHESE CODIR
# -------------------------------
summary_text = f"""
CA total : {ca_total:.2f} €
Nombre de produits représentant 80% du CA : {len(top_20_percent)}
Produits avec stock critique (<1 mois) : {len(risque_stock)}
Produits avec prix aberrants : {len(outliers_price)}
% produits sans liaison ERP : {df['a_valider_liaison'].mean()*100:.1f}%
"""

# Affichage rapide dans le notebook
print(summary_text)
print("\nTop 10 produits par CA :\n", top10)
print("\nExemples produits stock critique :\n", stock_critique)
print("\nExemples produits prix aberrants :\n", prix_aberrants)
print("\nProduits Web sans ERP :\n", web_sans_erp)
