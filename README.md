# project_ornithoScope

# Configuration
Ajouter dans le src/config.json le path vers les csv_base_path pour accéder aux images, ainsi que le path vers le csv de base de données.

# Setup venv
```
cd project_ornithoscope
python3 -m venv venv_ornithoscope
source venv_ornithoscope/bin/activate
pip install -r requirements.txt

```

# Setup organisation des dossiers

<img src="src/data/img_for_readme/tree0.png"
     alt=""
     style=""/>

</br>

# Historique
Problématique métier : détection d'oiseaux assisté par Deep Learning déployé sur Raspberry Pi.

- 1ère approche : EfficientDet0

    Analyse de la base de données.

    Mise en place d'une solution à l'état de l'art.

    Développement d'un modèle EfficientDet0. Split selon les tâches (différence de jour et différence de localisation)
    
    Beaucoup de complication liées aux librairies et aux dépendances. Pas assez de customisation.

    **F1 EfficientDet0 = 0.818**

- 2ème approche : MobileNetv1 et YOLOV2

    Changement de modèle, MobileNetv1 et YOLOV2. Split selon les tâches.

    Facilité d'utilisation, peu de dépendances, très customisable.

    **F1 MobileNetv1 et YOLOV1= 0.899**


# Améliorations possibles :

- Vérifier l'absence de dépendance entre la façon de split et le F1. **F1 random split = 0.893**

- Tester en retirant les deux espèces qui ne sont pas des oiseaux (campagnol et écureuil).

- Contrer le manque d'équilibre des classes : 
    
    Capé le nombres d'objets par classes, avec random split ou non. **F1 caped labels 300 = 0.823**, **F1 random split caped labels 300 = 0.653**, **F1 capted labels 100=0.791**
    
    Augmenter le nombre d'objets par classes. 

- Faire de l'augmentation de données (chaque image est utilisée de plusieurs manière via des rotations et autre)

- Tester différents autres modèles (MobileNetV2, V1small etc ...)

- Utilisation d'un modèle pré-entraîné sur un base de données d'oiseaux (iNaturalist) : 

    Déjà fait par Google mais impossible de l'utiliser, on a accès au modèle entier mais impossible d'extraire les poids. 

    Disponibilité des poids avec InceptionV3 sur TFHub mais problème de types de données.

    Mise en place de l'entraînement d'un modèle sur l'ensemble des données des oiseaux d'iNat, pour extraction des poids et amélioration de notre modèle.

    Possibilité de l'utiliser pour n'importe quel clade par la suite. Mise en place d'un pipeline avec sélection du clade à entraîner. 



# Tests effectués

Différents batchsizes (1 et 4)
Différents train time (1 et 4)

# Lignes de commandes utiles
```
export CUDA_VISIBLE_DEVICES=0,1
CUDA_VISIBLE_DEVICES=0, python3 train.py
CUDA_VISIBLE_DEVICES=1, python3 evaluate.py
```
