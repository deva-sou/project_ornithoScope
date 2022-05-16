# Rapport


# Sommaire
1- Etat de l'art 
2- Mat&Met
3- Résultats 
4- Discussion

# 1- Etat de l'art

# 2- Mat&Met

## 2.1- Base de données et contraintes
## 2.2- Utilisation de MobilenetV1 et YOLOv2 
## 2.3- Problématiques du déséquilibre de classes
### 2.3.1- Cap300
### 2.3.2- Data Augmentation optimisé par Google Brain Team -> utilisation de la V2


# 3- Résultats
## Présentation de la facçon de calcul des métriques (P,R,F1-score)
## Résultats du cap300 v2

# 4- Discusion
## 4.1- Résultats satisfaisants
## 4.2- Amélioration de la DB
Parler du data cleaning. 
Entrainement sur l'ensemble de train et de test. 
Test sur le tout et extraction des false positif et false negatif : concerne 180 images sur les 10 000. 
Identification du problème : mauvaise prédiction du modèle du à quoi ? Possibilité que l'oiseau soit mal labélisé, possibilité que la bbox soit mal tracée. 
Récupération d'images labélisées pour pouvoir améliorer les données
## 4.3- Tentative d'utilisationd d'iNaturalist

