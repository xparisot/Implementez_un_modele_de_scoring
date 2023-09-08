#!/usr/bin/env python
# coding: utf-8

# Import des librairies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.stats as stats


#Fonction de comptage de lignes et de colonne du dataset, affichage d'une representation graphique des sonnées manquantes
def valeurs(dataset):

    #Affichage du nombre de lignes et de colonnes
    lignes = len(dataset.index)
    colonnes = len(dataset.columns)
    print('Le nombre de lignes du fichier est de ', lignes, 'et de son nombre de colonnes est de', colonnes)

    #Representation graphique des données manquantes
    sns.set(rc = {'figure.figsize':(20,10)})
    sns.heatmap(dataset.isnull(), cbar=False)
    plt.rcParams.update({'font.size': 30})
    plt.tight_layout()
    plt.title('Representation graphique des valeurs manquantes', size=15)
    plt.plot()

def manquant (data):
    #Affichage des colonnes pour lesquelles il manque des données en pourcentage
    nb_na_sum = data.isnull().sum()
    round(nb_na_sum[nb_na_sum>0]*100,2).sort_values(ascending = True)
    print(nb_na_sum)
    #Affichage des colonnes pour lesquelles il manque des données en pourcentage
    nb_na = data.isnull().mean()
    round(nb_na[nb_na>0]*100,2).sort_values(ascending = True)
    print(nb_na)
    #Taux de remplissage moyen
    vnulles = data.isnull().sum().sum()
    nb_donnees_tot = np.product(data.shape)
    pourcentage_valeurs = round(data.isna().sum().sum()/(data.size)*100,2)
    print('Le jeu de données contient', vnulles, 'valeurs manquantes pour ',nb_donnees_tot, 'valeurs, soit', pourcentage_valeurs,'%')

#Format des données
def format_data(data):
    #On affiche le format des données
    print(data.shape[0],"produits")
    print(data.shape[1],"variables")
    print("\nType des variables:\n", data.dtypes.value_counts())
    #On affiche un graphique
    data.dtypes.value_counts().plot.pie()
    plt.title('Répartition du type de variable', size=15);

#Affichage de la proportion de nan dans un graphique

def proportion_nan(data):
    prop_nan = data.isna().sum().divide(data.shape[0]/100).sort_values(ascending=False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 30))
    ax = sns.barplot(y = prop_nan.index, x=prop_nan.values)
    ax.xaxis.set_ticks_position('top')
    plt.title('Quantité de données manquantes par colonne dans le jeu de données (en %)', size=15)
    plt.show()

    # On en profite pour regarder la distribution des Nan
    sns.set(style="whitegrid")
    ax = sns.distplot(prop_nan.values)
    ax.xaxis.set_ticks_position('top')
    plt.title('Répartition de la quantité de NaN', size=15)
    plt.show()

def doublons(data, colonne):
    # Ne pas oublier les '' dans le nom de la colonne
    data_doublons = data.loc[:,colonne:].columns
    data_doub = data.duplicated(subset=data_doublons, keep='first').value_counts()
    return data_doub

def suppr_doublons(data, colonne):
    # Ne pas oublier les '' dans le nom de la colonne
    data_doublons = data.loc[:,'product_name':].columns
    data = data[~data.duplicated(subset=data_doublons, keep='first')]

#Suppression des lignes contentant des outliers
def outliers_data(data, valeur):

    outliers_data = data[data[valeur] > 2000000].index
    data.drop(outliers_data, inplace=True)

    return data[data[valeur] > 2000000]

def outliers_neg(data, valeur):

    outliers_data = data[data[valeur] < 0].index
    data.drop(outliers_data, axis = 0, inplace=True)

    return data[data[valeur] < 0]

# fonction pour afficher le table de description statistique
def stat_descriptives(data, variables):
    warnings.filterwarnings('ignore')
    """
    Statistiques descriptives moyenne, mediane, variance, écart-type,
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
        liste_mean.append(data[col].mean())
        liste_median.append(data[col].median())
        liste_var.append(data[col].var(ddof=0))
        liste_std.append(data[col].std(ddof=0))
        liste_skew.append(data[col].skew())
        liste_kurtosis.append(data[col].kurtosis())
        liste_cols.append(col)
        liste_mode.append(data[col].mode().to_string())
        liste_min.append(data[col].min())
        liste_max.append(data[col].max())

    data_stats = [liste_mean, liste_median, liste_var, liste_std, liste_skew,
                  liste_kurtosis, liste_mode, liste_min, liste_max]
    df_stat = pd.DataFrame(data_stats, columns=liste_cols)

    return df_stat.style.hide_index()

def distibution(dataset):
    warnings.filterwarnings('ignore')

    sns.set_context("talk") # scaling automatique selon type de présentation ('notebook', 'paper', 'talk', 'poster')

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(21,30))

    sub = 0
    for i in range(len(dataset)):
        fig.add_subplot(len(dataset)//3+1,3,i+1)

        left, width = 0, 1
        bottom, height = 0, 1
        right = left + width
        top = bottom + height

        colonne = dataset[i]
        kstest = stats.kstest(data[colonne].notnull(),'norm')
        ax = sns.distplot(data[colonne], fit=stats.norm, kde=False)
        ax.set_title("Distribution vs loi normale : {}".format(colonne))
        ax.text(right, top, 'Test Kolmogorov-Smirnov \n Pvalue: {:.2} \n Stat: {:.2}'.format(kstest.pvalue, kstest.statistic),
                horizontalalignment='right',
                verticalalignment='top',
                style='italic', transform=ax.transAxes, fontsize = 12,
                bbox={'facecolor':'#00afe6', 'alpha':0.5, 'pad':0})
        sub += 1

        fig.tight_layout()

    plt.show()

def display_factorial_planes(   X_projected,
                                x_y,
                                pca=None,
                                labels = None,
                                clusters=None,
                                alpha=1,
                                figsize=[10,8],
                                marker="." ):
    """
    Affiche la projection des individus

    Positional arguments :
    -------------------------------------
    X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

    Optional arguments :
    -------------------------------------
    pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
    labels : list ou tuple : les labels des individus à projeter, default = None
    clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
    alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8]
    marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
    """

    # Transforme X_projected en np.array
    X_ = np.array(X_projected)

    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize:
        figsize = (7,6)

    # On gère les labels
    if  labels is None :
        labels = []
    try :
        len(labels)
    except Exception as e :
        raise e

    # On vérifie la variable axis
    if not len(x_y) ==2 :
        raise AttributeError("2 axes sont demandées")
    if max(x_y )>= X_.shape[1] :
        raise AttributeError("la variable axis n'est pas bonne")

    # on définit x et y
    x, y = x_y

    # Initialisation de la figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # On vérifie s'il y a des clusters ou non
    c = None if clusters is None else clusters

    # Les points
    # plt.scatter(   X_[:, x], X_[:, y], alpha=alpha,
    #                     c=c, cmap="Set1", marker=marker)
    sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=c)

    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe
    if pca :
        v1 = str(round(100*pca.explained_variance_ratio_[x]))  + " %"
        v2 = str(round(100*pca.explained_variance_ratio_[y]))  + " %"
    else :
        v1=v2= ''

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f"F{x+1} {v1}")
    ax.set_ylabel(f"F{y+1} {v2}")

    # Valeur x max et y max
    x_max = np.abs(X_[:, x]).max() *1.1
    y_max = np.abs(X_[:, y]).max() *1.1

    # On borne x et y
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom= -y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
    plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)

    # Affichage des labels des points
    if len(labels) :
        for i,(_x,_y) in enumerate(X_[:,[x,y]]):
            plt.text(_x, _y+0.05, labels[i], fontsize='14', ha='center',va='center')

    # Titre et display
    plt.title(f"Projection des individus (sur F{x+1} et F{y+1})")
    plt.show()
