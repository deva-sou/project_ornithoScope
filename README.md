<h1 align="center">Project OrnithoScope</h3>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of content</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Virtual environment</a></li>
        <li><a href="#installation">Folder organization</a></li>
      </ul>
    </li>
    <li><a href="#how-it-works">How it works</a></li>
    <li><a href="#how-to-use-it">How to use it</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About the project
Developed for the ECONNECT project (CNRS-LEFE), OrnithoScope ensures the automatic detection of birds by Deep Learning. Here we developed an algorithm based on MobileNet and YOLO for Raspberry PI.


<!-- GETTING STARTED -->
# Getting Started
## Prerequisites
Depending on your OS, GPU and Drivers, you will need a custom install to use this algorithm. With at least:
* python3
* numpy
* pandas
* keras
* sklearn
* tensorflow
* bbaug

## Virtual environment
For that, you can use a virtual environment with [python venv](https://docs.python.org/3/library/venv.html)
```
cd project_ornithoscope
python3 -m venv venv_ornithoScope
source venv_ornithoscope/bin/activate
# optionnal: (depending on your hardware) 
# pip install -r requirements.txt

```
## Folder organization
We recommend you to get the data folder and project folder (with the virtual environment) near by.

<div style="text-align: center;">
  <img src="src/data/imgs/img_for_readme/tree0.png" alt="tree">
</div>

<!-- HOW IT WORK -->
# How it works
## Code source folder (project_ornithoscore/src)
In src folder you will find: 
* [train.py](#trainpy) (for training)
* [evaluate.py](#evaluatepy) (for evaluation)
* [predict.py](#predictpy) (for prediction)
### train.py
### evaluate.py
### predict.py

## YOLOV2 folder (project_ornithoscore/src/keras_yolov2)
In this YOLO folder you will find: 
* [backend.py](#backendpy) (to create backend of models)
* [frontend.py](#frontendpy) (to create frontend of models)
* [preprocessing.py](#preprocessingpy) (for preprocessing)
* [map_evaluation.py](#mapevaluationpy) (for evaluation)
* [utils.py](#utilspy) (for utilitarian functions)

### backend.py
### frontend.py
### preprocessing.py
### map_evaluation.py
### utils.py

<!-- HOW TO USE IT -->
# How to use it
```
python3 train.py -c path_custom_config.json
python3 evaluate.py -c path_custom_config.json
python3 predict.py -c path_custom_config.json -w path_seleccted_weights.h5 -i path_image_folder_to_predict
```

<!-- CONTACT -->
# Contact
SOU Deva [@devasou](https://www.linkedin.com/in/deva-sou/)
