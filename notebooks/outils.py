#%%
from datetime import timedelta

# importation des librairies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import scipy.stats as stats
from jupyterlab_widgets import data


# Affichage des informations du dataframe
def info(df):

    display(df.head(3))
    print(f'Taille :-------------------------------------------------------------- {df.shape}')
    print("--"*50)
    print(f'Types :{df.dtypes.value_counts()}')
    print("--"*50)
    print(f'Types :{df.dtypes}')
    print("--"*50)
    print("Valeurs manquantes par colonnes (%): ")
    print((((df.isna().sum()/df.shape[0])*100).round(2)).sort_values(ascending=False))
    print("--"*50)
    print("Valeurs différentes par variables : ")
    for cols in df:
        if df[cols].nunique() < 30:
            print(f'{cols :-<70} {df[cols].unique()}')
        else:
            print(f'{cols :-<70} contient {df[cols].nunique()} valeurs différentes')
    print("--"*50)
    print(f"Nombre de doublons : {df.duplicated().sum()}")


# Fonction de comptage de lignes et de colonne du dataset, affichage d'une representation\
# graphique des sonnées manquantes

def valeurs(df):

    # Affichage du nombre de lignes et de colonnes
    lignes = len(df.index)
    colonnes = len(df.columns)
    print('Le nombre de lignes du fichier est de ', lignes, 'et de son nombre de colonnes est de', colonnes)

    # Representation graphique des données manquantes
    sns.set(rc={'figure.figsize': (20, 10)})
    sns.heatmap(df.isnull(), cbar=False)
    plt.rcParams.update({'font.size': 30})
    plt.tight_layout()
    plt.title('Representation graphique des valeurs manquantes', size=15)
    plt.plot()


def manquant(df):

    # Affichage des colonnes pour lesquelles il manque des données en pourcentage
    nb_na_sum = df.isnull().sum()
    round(nb_na_sum[nb_na_sum > 0]*100, 2).sort_values(ascending=True)
    print(nb_na_sum)
    # Affichage des colonnes pour lesquelles il manque des données en pourcentage
    nb_na = df.isnull().mean()
    round(nb_na[nb_na > 0]*100, 2).sort_values(ascending=True)
    print(nb_na)
    # Taux de remplissage moyen
    nulles = df.isnull().sum().sum()
    nb_donnees_tot = np.product(df.shape)
    pourcentage_valeurs = round(df.isna().sum().sum() / df.size * 100, 2)
    print('Le jeu de données contient', nulles, 'valeurs manquantes pour ', nb_donnees_tot, 'valeurs, soit',
          pourcentage_valeurs, '%')


# Format des données
def format_data(df):

    # On affiche le format des données
    print(df.shape[0], "produits")
    print(df.shape[1], "variables")
    print("\nType des variables:\n", df.dtypes.value_counts())
    # On affiche un graphique
    df.dtypes.value_counts().plot.pie()
    plt.title('Répartition du type de variable', size=15)
    plt.plot()


def doublons(df, colonne):
    # Ne pas oublier les '' dans le nom de la colonne
    data_doublons = df.loc[:, colonne:].columns
    data_double = df.duplicated(subset=data_doublons, keep='first').value_counts()
    return data_double


# fonction pour afficher la table de description statistique
def stat_descriptives(df, variables):
    warnings.filterwarnings('ignore')
    """
    Statistiques descriptives moyenne, médiane, variance, écart-type,
    skewness et kurtosis du dataframe transmis en paramètre
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                liste_variables : colonne dont on veut voir les stat descr
    @param OUT : dataframe des statistiques descriptives
    """
    liste_mean = ['mean']
    liste_median = ['median']
    liste_var = ['var']
    liste_std = ['std']
    liste_skew = ['skew']
    liste_kurtosis = ['kurtosis']
    liste_mode = ['mode']
    liste_cols = ['Desc']
    liste_max = ['Max']
    liste_min = ['Min']

    for col in variables:
        liste_mean.append(df[col].mean())
        liste_median.append(df[col].median())
        liste_var.append(df[col].var(ddof=0))
        liste_std.append(df[col].std(ddof=0))
        liste_skew.append(df[col].skew())
        liste_kurtosis.append(df[col].kurtosis())
        liste_cols.append(col)
        liste_mode.append(df[col].mode().to_string())
        liste_min.append(df[col].min())
        liste_max.append(df[col].max())

    data_stats = [liste_mean, liste_median, liste_var, liste_std, liste_skew,
                  liste_kurtosis, liste_mode, liste_min, liste_max]
    df_stat = pd.DataFrame(data_stats, columns=liste_cols)

    return df_stat.style.hide_index()


def distribution(dataset):
    warnings.filterwarnings('ignore')

    sns.set_context("talk")  # scaling automatique selon type de présentation ('notebook', 'paper', 'talk', 'poster')

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(21, 30))

    sub = 0
    for i in range(len(dataset)):
        fig.add_subplot(len(dataset)//3+1, 3, i+1)

        left, width = 0, 1
        bottom, height = 0, 1
        right = left + width
        top = bottom + height

        colonne = dataset[i]
        kstest = stats.kstest(data_train[colonne].notnull(), 'norm')
        ax = sns.distplot(data_train[colonne], fit=stats.norm, kde=False)
        ax.set_title("Distribution vs loi normale : {}".format(colonne))
        ax.text(right, top, 'Test Kolmogorov-Smirnov \n Pvalue: {:.2} \n Stat: {:.2}'.format(kstest.pvalue,
                                                                                             kstest.statistic),
                horizontalalignment='right',
                verticalalignment='top',
                style='italic', transform=ax.transAxes, fontsize=12,
                bbox={'facecolor': '#00afe6', 'alpha': 0.5, 'pad': 0})
        sub += 1

        fig.tight_layout()

    plt.show()


def calculate_rfm(df, period, today):
    """
    Calcul des valeurs R, F et M.

    Paramètres:
    data(pd.DataFrame): colonnes 'payment_value','purchase_time' et 'customer_id'
    period(int): Nombre de jours
    today(dt.datetime): date de fin de période

    Return:
    rfm(pd.DataFrame): retourne un pd.DataFrame avec les variables Recency,
    Frequency et Monetary
    """

    df_rfm = df.loc[df["order_purchase_timestamp"] >= today - timedelta(days=period)]
    # Nombre de jours passés entre aujourd'hui
    # et la dernière commande de chaque utilisateur
    df_rfm.loc[:, "DaysSinceOrder"] = df_rfm.loc[:, "order_purchase_timestamp"] \
        .map(lambda d: (today - d).days)

    aggr = {
        'DaysSinceOrder': lambda x: x.min(),
        # Nombre de jours depuis la dernière commande (Recency)
        'order_purchase_timestamp': lambda x:
        len([d for d in x if d >= today - timedelta(days=period)]),
        # le nombre total de transactions sur la période passée (Frequency)
    }

    rfm = df_rfm.groupby('customer_unique_id').agg(aggr).reset_index()
    rfm.rename(columns={'DaysSinceOrder': 'Recency',
                        'order_purchase_timestamp': 'Frequency'},
               inplace=True)

    # Montant total de toutes les transactions sur la période définie
    rfm['Monetary'] = rfm.loc[:, 'customer_unique_id'].apply(lambda x:
                                                             df_rfm.loc[df_rfm['customer_unique_id'] ==
                                                                        x, 'payment_value'].sum())

    # Affichage des 5 premières lignes
    rfm.head()

    return rfm


def viz_rfm(rfm, figsize=(20, 20)):

    sns.set_style("whitegrid")
    plt.figure(1, figsize=figsize)

    for i, col in enumerate(rfm):
        plt.subplot(len(rfm.columns), 1, i + 1)
        plt.hist(rfm[col],
                 bins=int(1 + np.log2(len(rfm))),
                 label='skweness : ' + str(round(rfm[col].skew(), 2)),
                 density=True)
        plt.title('Representation RFM', size=15)
        plt.ylabel("Nombre de clients (%)")
        plt.xlabel(col)
        plt.legend()
    plt.show()


def r_score(x, quintiles):
    """
    Attribut une note entre 1 et 5 suivant l'emplacement de x par
    rapport aux quintiles.
    x(float): valeur
    quintiles(dict): voir fonction calculQuantile
    """

    if x <= quintiles['Recency'][.25]:
        return 4
    elif x <= quintiles['Recency'][.50]:
        return 3
    elif x <= quintiles['Recency'][.75]:
        return 2
    else:
        return 1


def fm_score(x, quintiles, c):
    """
    Attribut une note entre 1 et 5 suivant l'emplacement de x par
    rapport aux quintiles.
    x(float): valeur
    quintiles(dict): voir fonction calculQuantile
    """

    if x <= quintiles[c][.25]:
        return 1
    elif x <= quintiles[c][.50]:
        return 2
    elif x <= quintiles[c][.75]:
        return 3
    else:
        return 4


def calculate_quantile(rfm):
    """
    Calcul des quantiles pour chaque variable Recency, Frequency et Monetary
    puis attribut un score (fm_score et r_score) suivant la valeur
    des variables de chaque individu
    rfm(pd.DataFrame): présente les colonnes 'Recency', 'Frequency' et Monetary
    """
    quintiles = rfm[['Recency',
                     'Frequency_sf_Log',
                     'Monetary_sf_Log']].quantile([.25, .50, .75]).to_dict()

    rfm['R'] = rfm['Recency'].apply(lambda x:
                                    r_score(x, quintiles))
    rfm['F'] = rfm['Frequency_sf_Log'].apply(lambda x:
                                             fm_score(x, quintiles, 'Frequency_sf_Log'))
    rfm['M'] = rfm['Monetary_sf_Log'].apply(lambda x:
                                            fm_score(x, quintiles, 'Monetary_sf_Log'))
    return rfm


def segmentation(rfm):
    """
    Retourne le segment associé au client en fonction de son score pour les
    variables R, F et M

    Paramètres:
    rfm(pd.DataFrame): doit contenir les colonnes R, F et M

    Return:
    str: Nom du segment
    """

    if 3 <= rfm["R"] <= 4 and 3 <= rfm["F"] <= 4 and 3 <= rfm["M"] <= 4:
        return "Très bons clients"

    elif 3 <= rfm["R"] <= 4 and 3 <= rfm["F"] <= 4 and 1 <= rfm["M"] <= 2:
        return "Petits clients fidèles"

    elif 3 <= rfm["R"] <= 4 and 1 <= rfm["F"] <= 2 and 3 <= rfm["M"] <= 4:
        return "Bons clients à potentiel"

    elif 1 <= rfm["R"] <= 2 and 3 <= rfm["F"] <= 4 and 3 <= rfm["M"] <= 4:
        return "Très bons clients à risque de départ"

    elif 3 <= rfm["R"] <= 4 and 1 <= rfm["F"] <= 2 and 1 <= rfm["M"] <= 2:
        return "Clients à faible potentiel"

    elif 1 <= rfm["R"] <= 2 and 3 <= rfm["F"] <= 4 and 1 <= rfm["M"] <= 2:
        return "Petits clients fidèles à risque de départ"

    elif 1 <= rfm["R"] <= 2 and 1 <= rfm["F"] <= 2 and 3 <= rfm["M"] <= 4:
        return "Clients à potentiel"
    else:
        return "Clients occasionnels à faible potentiel"

