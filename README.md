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

<img src="src/data/imgs/img_for_readme/tree0.png"
     alt=""
     style=""/>

</br>

# Exécution des fichiers
```
python3 train.py -c path_custom_config.json
python3 evaluate.py -c path_custom_config.json
python3 predict.py -c path_custom_config.json -w path_seleccted_weights.h5 -i path_image_folder_to_predict
```
