#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notebook Python pour CODIR
Analyse ventes, gestion stock, export graphiques PNG et HTML interactifs (Plotly)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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

full = (
    web.merge(liaison[['ref_web_norm', 'ref_erp_norm']], how='outer', on='ref_web_norm')
       .merge(erp, how='left', on='ref_erp_norm', suffixes=('_web', '_erp'))
)

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
if qty_col in df.columns and price_col in df.columns:
    df['total_sales_value'] = df[qty_col] * df[price_col]
    ca_total = df['total_sales_value'].sum()
else:
    raise ValueError("Colonnes pour qty ou price manquantes. CA non calculable.")

# -------------------------------
# CA DU MOIS D’OCTOBRE
# -------------------------------
date_col_candidates = [col for col in df.columns if 'date' in col.lower()]
if date_col_candidates:
    date_col = date_col_candidates[0]
    print(f"✅ Colonne de date détectée : {date_col}")
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df_october = df[df[date_col].dt.month == 10]
    
    if not df_october.empty:
        ca_october = df_october['total_sales_value'].sum()
        print(f"💰 Chiffre d'affaires du mois d'octobre : {ca_october:,.2f} €")
    else:
        print("⚠️ Aucune donnée trouvée pour le mois d'octobre.")
else:
    print("❌ Aucune colonne de date détectée dans le jeu de données.")

# --- Graphique CA mensuel Plotly ---
if date_col_candidates:
    df['month'] = df[date_col].dt.to_period('M')
    ca_mensuel = df.groupby('month')['total_sales_value'].sum().reset_index()
    ca_mensuel['month'] = ca_mensuel['month'].astype(str)

    fig_ca_mois = px.bar(
        ca_mensuel,
        x='month',
        y='total_sales_value',
        title="Chiffre d'affaires mensuel",
        labels={'month': 'Mois', 'total_sales_value': 'CA (€)'},
        color='total_sales_value',
        color_continuous_scale='Blues'
    )
    fig_ca_mois.update_layout(xaxis_tickangle=-45, template='plotly_white')
    fig_ca_mois.add_hline(
        y=ca_october if 'ca_october' in locals() else 0,
        line_dash="dot",
        annotation_text="CA Octobre",
        annotation_position="top left"
    )
    fig_ca_mois.write_html(plots_path / "ca_mensuel.html")
    print(f"📊 Graphique CA mensuel sauvegardé : {plots_path / 'ca_mensuel.html'}")

# -------------------------------
# ANALYSES SUPPLÉMENTAIRES
# -------------------------------
df_sorted = df.sort_values(by='total_sales_value', ascending=False)
df_sorted['ca_cumsum'] = df_sorted['total_sales_value'].cumsum()
df_sorted['ca_cumperc'] = df_sorted['ca_cumsum'] / ca_total * 100
top_20_percent = df_sorted[df_sorted['ca_cumperc'] <= 80]

# Outliers prix
df['price_zscore'] = np.abs(stats.zscore(df[price_col].fillna(0)))
Q1 = df[price_col].quantile(0.25)
Q3 = df[price_col].quantile(0.75)
IQR = Q3 - Q1
df['price_outlier_iqr'] = ((df[price_col] < Q1 - 1.5 * IQR) | (df[price_col] > Q3 + 1.5 * IQR))
outliers_price = df[(df['price_zscore'] > 3) | (df['price_outlier_iqr'])]

# Marges et rotation stock
df['margin_rate'] = ((df[price_col] - df['purchase_price_num']) / df[price_col] * 100).round(2)
df['rotation_stock'] = df['total_sales_value'] / df['stock_quantity_num'].replace(0, np.nan)
df['mois_stock'] = df['stock_quantity_num'] / df['total_sales_value'].replace(0, np.nan)
risque_stock = df[df['mois_stock'] < 1]

# Tableaux pour synthèse
top10 = df_sorted[['post_title', 'total_sales_value']].head(10)
prix_aberrants = outliers_price[['post_title', price_col, 'price_zscore']].head(10)
stock_critique = risque_stock[['post_title', 'stock_quantity_num', 'total_sales_value', 'mois_stock']].head(10)
web_sans_erp = df[df['a_valider_liaison']][['post_title', 'total_sales', 'price_num']]

# -------------------------------
# FONCTIONS D'EXPORT
# -------------------------------
def save_plot_png(fig, filename, tight=True, dpi=150):
    fig.savefig(filename, bbox_inches='tight' if tight else None, dpi=dpi)
    plt.close(fig)
    return str(filename)

def save_plot_html(fig, filename):
    fig.write_html(filename)
    print(f"Graphique interactif sauvegardé : {filename}")

# -------------------------------
# GRAPHIQUES EXISTANTS (Matplotlib + Plotly)
# -------------------------------
top_n = 50
df_topn = df_sorted.head(top_n).copy()
df_topn['ca_cumsum'] = df_topn['total_sales_value'].cumsum()
df_topn['ca_cumperc'] = df_topn['ca_cumsum'] / ca_total * 100

# --- 1️⃣ Top 50 produits ---
fig1, ax = plt.subplots(figsize=(14, 7))
ax.bar(range(len(df_topn)), df_topn['total_sales_value'], color='skyblue', label='CA produit')
ax2 = ax.twinx()
ax2.plot(range(len(df_topn)), df_topn['ca_cumperc'], color='red', marker='o', label='CA cumulé (%)')
ax2.axhline(80, color='orange', linestyle='--', label='80% du CA')

labels = df_topn['post_title'] if 'post_title' in df_topn.columns else df_topn['ref_web_norm']
ax.set_xticks(range(len(df_topn)))
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_xlabel("Produits triés par CA décroissant", fontsize=12)
ax.set_ylabel("CA produit (€)", fontsize=12)
ax2.set_ylabel("CA cumulé (%)", fontsize=12)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title("Top 50 produits et CA cumulatif", fontsize=16, fontweight='bold')
save_plot_png(fig1, plots_path / "top50_ca.png")

# Version Plotly interactive
fig1_plotly = go.Figure()
fig1_plotly.add_trace(go.Bar(x=df_topn['post_title'], y=df_topn['total_sales_value'], name="CA produit", marker_color='skyblue'))
fig1_plotly.add_trace(go.Scatter(x=df_topn['post_title'], y=df_topn['ca_cumperc'], name="CA cumulé (%)", mode='lines+markers', yaxis='y2'))
fig1_plotly.update_layout(
    title="Top 50 produits et CA cumulatif",
    xaxis_title="Produit",
    yaxis_title="CA (€)",
    yaxis2=dict(title="CA cumulé (%)", overlaying='y', side='right'),
    template='plotly_white'
)
save_plot_html(fig1_plotly, plots_path / "top50_ca.html")

# --- 2️⃣ Boxplot des prix ---
fig2, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x=df['price_num'], ax=ax)
ax.set_title("Boxplot des prix produits", fontsize=16, fontweight='bold')
ax.set_xlabel("Prix (€)", fontsize=14)
save_plot_png(fig2, plots_path / "boxplot_prix.png")

fig2_plotly = px.box(df, x='price_num', title="Boxplot des prix produits")
save_plot_html(fig2_plotly, plots_path / "boxplot_prix.html")

# --- 3️⃣ Distribution des marges ---
fig3, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df['margin_rate'].dropna(), bins=20, kde=True, ax=ax, color='skyblue')
ax.set_title("Distribution des taux de marge (%)", fontsize=16, fontweight='bold')
save_plot_png(fig3, plots_path / "hist_marge.png")

fig3_plotly = px.histogram(df, x='margin_rate', nbins=20, title="Distribution des taux de marge (%)", marginal="box", color_discrete_sequence=['skyblue'])
save_plot_html(fig3_plotly, plots_path / "hist_marge.html")

# --- 4️⃣ Mois de stock ---
fig4, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df['mois_stock'].dropna(), bins=20, kde=True, ax=ax, color='orange')
ax.set_title("Distribution des mois de stock", fontsize=16, fontweight='bold')
save_plot_png(fig4, plots_path / "hist_stock.png")

fig4_plotly = px.histogram(df, x='mois_stock', nbins=20, title="Distribution des mois de stock", color_discrete_sequence=['orange'])
save_plot_html(fig4_plotly, plots_path / "hist_stock.html")

# --- 5️⃣ Corrélations ---
corr = df[['price_num', 'purchase_price_num', 'stock_quantity_num', 'total_sales_value', 'margin_rate']].corr()
fig5, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title("Corrélations entre variables clés", fontsize=16, fontweight='bold')
save_plot_png(fig5, plots_path / "heatmap_corr.png")

fig5_plotly = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', title="Corrélations entre variables clés")
save_plot_html(fig5_plotly, plots_path / "heatmap_corr.html")

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

print(summary_text)
print("\nTop 10 produits par CA :\n", top10)
print("\nExemples produits stock critique :\n", stock_critique)
print("\nExemples produits prix aberrants :\n", prix_aberrants)
print("\nProduits Web sans ERP :\n", web_sans_erp)
