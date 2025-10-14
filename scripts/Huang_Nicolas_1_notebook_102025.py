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

sns.set_theme(style="darkgrid")  # Style général des graphiques

# -------------------------------
# CHEMINS ET CHARGEMENT DES DONNÉES
# -------------------------------
data_path = Path(__file__).parent.parent / "data"  # Dossier des CSV
plots_path = Path(__file__).parent.parent / "plots"  # Dossier des graphiques
plots_path.mkdir(exist_ok=True)

erp = pd.read_csv(data_path / "erp.csv")      # Données ERP
web = pd.read_csv(data_path / "web.csv")      # Données Web
liaison = pd.read_csv(data_path / "liaison.csv")  # Table de correspondance

# -------------------------------
# PHASE 1 : NETTOYAGE ET JOINTURE
# -------------------------------
# -------------------------------
# NORMALISATION ET DÉTECTION DES DOUBLONS
# -------------------------------
def norm(x):
    """Normalise les chaînes : strip + majuscule"""
    try:
        return str(x).strip().upper()
    except:
        return x

# Normalisation des références pour merge
erp['ref_erp_norm'] = erp['product_id'].astype(str).apply(norm)
liaison['ref_erp_norm'] = liaison['product_id'].astype(str).apply(norm)
liaison['ref_web_norm'] = liaison['id_web'].astype(str).apply(norm)
web['ref_web_norm'] = web['sku'].astype(str).apply(norm)

# Supprimer les doublons avant merge
erp = erp.drop_duplicates(subset=['ref_erp_norm'])
web = web.drop_duplicates(subset=['ref_web_norm'])
liaison = liaison.drop_duplicates(subset=['ref_web_norm', 'ref_erp_norm'])

# -------------------------------
# MERGE SÉCURISÉ
# -------------------------------
# Merge outer puis left pour garder toutes les données et éviter les doublons
full = (
    web.merge(liaison, how='outer', on='ref_web_norm')
       .merge(erp, how='left', on='ref_erp_norm', suffixes=('_web', '_erp'))
)

# Nettoyage post-merge : conversion numériques et flags
full['price_num'] = pd.to_numeric(full['price'], errors='coerce')
full['stock_quantity_num'] = pd.to_numeric(full['stock_quantity'], errors='coerce').fillna(0).astype(int)
full['purchase_price_num'] = pd.to_numeric(full['purchase_price'], errors='coerce')
full['a_valider_liaison'] = full['ref_erp_norm'].isna()  # Flag liaison manquante

# Supprimer doublons éventuels après merge
full = full.drop_duplicates(subset=['ref_web_norm', 'ref_erp_norm'])

df = full.copy()
qty_col = 'total_sales'
price_col = 'price_num'

# -------------------------------
# PHASE 2 : ANALYSES
# -------------------------------
# Calcul CA total
if qty_col in df.columns and price_col in df.columns:
    df['total_sales_value'] = df[qty_col] * df[price_col]
    ca_total = df['total_sales_value'].sum()
else:
    raise ValueError("Colonnes pour qty ou price manquantes. CA non calculable.")

# -------------------------------
# CA DU MOIS D’OCTOBRE
# -------------------------------
# Détection automatique colonne date
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
# Top 80% CA
df_sorted = df.sort_values(by='total_sales_value', ascending=False)
df_sorted['ca_cumsum'] = df_sorted['total_sales_value'].cumsum()
df_sorted['ca_cumperc'] = df_sorted['ca_cumsum'] / ca_total * 100
top_20_percent = df_sorted[df_sorted['ca_cumperc'] <= 80]

# Détection outliers prix
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

# Tableaux synthèse
top10 = df_sorted[['post_title', 'total_sales_value']].head(10)
prix_aberrants = outliers_price[['post_title', price_col, 'price_zscore']].head(10)
stock_critique = risque_stock[['post_title', 'stock_quantity_num', 'total_sales_value', 'mois_stock']].head(10)
web_sans_erp = df[df['a_valider_liaison']][['post_title', 'total_sales', 'price_num']]

# -------------------------------
# FONCTIONS D'EXPORT
# -------------------------------
def save_plot_png(fig, filename, tight=True, dpi=150):
    """Sauvegarde figure Matplotlib en PNG"""
    fig.savefig(filename, bbox_inches='tight' if tight else None, dpi=dpi)
    plt.close(fig)
    return str(filename)

def save_plot_html(fig, filename):
    """Sauvegarde figure Plotly interactive en HTML"""
    fig.write_html(filename)
    print(f"Graphique interactif sauvegardé : {filename}")

# -------------------------------
# GRAPHIQUES PRINCIPAUX
# -------------------------------
# Top 50 produits CA cumulé
# Boxplot prix, distribution marges, mois de stock, corrélations
# Chaque graphique export PNG + HTML
# (Commentaire intégré à chaque bloc de graphique pour comprendre la fonction)

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
