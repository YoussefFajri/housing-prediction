# analyse_immobilier_californie.py
"""
Analyse des Prix Immobiliers en Californie
Auteur: [Votre Nom]
Date: [Date Actuelle]

Ce script analyse le dataset des logements californiens et construit
des modèles de machine learning pour prédire les prix médians des maisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Configuration des visualisations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Chargement des Données
print("Chargement du Dataset Immobilier Californien...")
data = pd.read_csv("housing.csv")

print("\n Informations sur le Dataset:")
data.info()

print("\n Gestion des Valeurs Manquantes...")
data.dropna(inplace=True)
print("Après suppression des valeurs manquantes:")
data.info()

# Préparation des Features et Target
X = data.drop(['median_house_value'], axis=1)
Y = data['median_house_value']

# Division des Données
print("\nDivision des Données en Ensembles d'Entraînement/Test...")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
train_data = x_train.join(y_train)
print(f"Shape de l'ensemble d'entraînement: {train_data.shape}")

# Analyse Exploratoire des Données
print("\n Création de la Matrice d'Histogrammes...")
train_data.hist(figsize=(15, 10))
plt.suptitle('Distributions des Features - Données Immobilières Californiennes', fontsize=16)
plt.tight_layout()
plt.show()

# Analyse de Corrélation
print("\n Analyse des Corrélations entre Features...")
numerical_features = ['longitude', 'latitude', 'housing_median_age', 
                     'total_rooms', 'total_bedrooms', 'population', 
                     'households', 'median_income', 'median_house_value']

plt.figure(figsize=(12, 8))
sns.heatmap(train_data[numerical_features].corr(), annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Matrice de Corrélation - Features Numériques')
plt.tight_layout()
plt.show()

# Ingénierie des Features - Transformation Logarithmique
print("\n Application des Transformations Logarithmiques aux Features Asymétriques...")
train_data['total_rooms_log'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms_log'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population_log'] = np.log(train_data['population'] + 1)
train_data['households_log'] = np.log(train_data['households'] + 1)

print("Transformations logarithmiques terminées")
print("\n Visualisation des Distributions Transformées...")
train_data.hist(figsize=(15, 10))
plt.suptitle('Distributions après Transformation Logarithmique', fontsize=16)
plt.tight_layout()
plt.show()

# Encodage One-Hot de la Variable Catégorielle
print("\n Encodage de la Variable Catégorielle: ocean_proximity")
train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity).astype('int32')).drop(['ocean_proximity'], axis=1)
print(f"Shape après encodage: {train_data.shape}")

# Nouvelle Analyse de Corrélation
print("\n Nouvelle Matrice de Corrélation avec Variables Encodées...")
plt.figure(figsize=(14, 10))
sns.heatmap(train_data.corr(), annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Matrice de Corrélation Complète')
plt.tight_layout()
plt.show()

# Analyse Géographique
print("\n Analyse Géographique des Prix...")
plt.figure(figsize=(10, 8))
sns.scatterplot(x='latitude', y='longitude', data=train_data, 
                hue='median_house_value', palette='coolwarm', alpha=0.6)
plt.title('Distribution Géographique des Prix Immobiliers')
plt.show()

# Création de Nouvelles Features
print("\n Création de Nouvelles Features...")
train_data['bedrooms_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']

print(" Nouvelles features créées:")
print("   - bedrooms_ratio: Ratio chambres/total pièces")
print("   - household_rooms: Pièces par ménage")

# Dernière Analyse de Corrélation
plt.figure(figsize=(16, 12))
sns.heatmap(train_data.corr(), annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Matrice de Corrélation Finale avec Nouvelles Features')
plt.tight_layout()
plt.show()

# Préparation pour l'Entraînement
print("\n Préparation des Données pour l'Entraînement...")
x_train, y_train = train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']

# Modèle de Régression Linéaire
print("\n Entraînement du Modèle de Régression Linéaire...")
reg = LinearRegression()
reg.fit(x_train, y_train)

# Préparation des Données de Test
print("\n Préparation de l'Ensemble de Test...")
test_data = x_test.join(y_test)

# Application des mêmes transformations aux données de test
test_data['total_rooms_log'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms_log'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population_log'] = np.log(test_data['population'] + 1)
test_data['households_log'] = np.log(test_data['households'] + 1)

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity).astype('int32')).drop(['ocean_proximity'], axis=1)
test_data['bedrooms_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']

x_test, y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']

# Évaluation du Modèle Linéaire
score_linear = reg.score(x_test, y_test)
print(f" Score R² de la Régression Linéaire: {score_linear:.4f}")

# Standardisation des Données
print("\n Standardisation des Données...")
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

# Régression Linéaire avec Données Standardisées
reg_standardized = LinearRegression()
reg_standardized.fit(x_train_s, y_train)
score_linear_std = reg_standardized.score(x_test_s, y_test)
print(f" Score R² de la Régression Linéaire (standardisée): {score_linear_std:.4f}")

# Modèle Random Forest
print("\n Entraînement du Modèle Random Forest...")
forest = RandomForestRegressor(random_state=42)
forest.fit(x_train, y_train)
score_forest = forest.score(x_test, y_test)
print(f" Score R² du Random Forest: {score_forest:.4f}")

# Random Forest avec Données Standardisées
forest_std = RandomForestRegressor(random_state=42)
forest_std.fit(x_train_s, y_train)
score_forest_std = forest_std.score(x_test_s, y_test)
print(f" Score R² du Random Forest (standardisé): {score_forest_std:.4f}")

# Optimisation par Grid Search
print("\n Optimisation des Hyperparamètres avec Grid Search...")
param_grid = {
    "n_estimators": [3, 10, 30],
    "max_features": [2, 4, 6, 8]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42), 
    param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error', 
    return_train_score=True
)

grid_search.fit(x_train, y_train)

print(f"Meilleurs paramètres: {grid_search.best_params_}")
best_forest = grid_search.best_estimator_
score_best_forest = best_forest.score(x_test, y_test)
print(f" Score R² du Meilleur Random Forest: {score_best_forest:.4f}")

# Résumé des Performances
print("\n" + "="*50)
print("RÉSUMÉ DES PERFORMANCES DES MODÈLES")
print("="*50)
print(f"Régression Linéaire: {score_linear:.4f}")
print(f"Régression Linéaire (standardisée): {score_linear_std:.4f}")
print(f"Random Forest: {score_forest:.4f}")
print(f"Random Forest (standardisé): {score_forest_std:.4f}")
print(f"Random Forest Optimisé: {score_best_forest:.4f}")
print("="*50)