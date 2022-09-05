path_to_true_cap300 = "/home/acarlier/code/project_ornithoScope/src/data/inputs/input_train_trainset_cap300.csv"
path_to_false_cap300 = "/home/acarlier/code/project_ornithoScope/src/data/inputs/input_train_cap300.csv"
path_to_validset = "/home/acarlier/code/project_ornithoScope/src/data/inputs/input_train_iNat_validset.csv"
path_to_testset = "/home/acarlier/code/project_ornithoScope/src/data/inputs/input_test.csv"
import pandas as pd
import numpy as np
from IPython.display import display

df_true = pd.read_csv(path_to_true_cap300)
df_false = pd.read_csv(path_to_false_cap300)
df_valid = pd.read_csv(path_to_validset)
df_test = pd.read_csv(path_to_testset)

img_path_true = df_true["task_2021_11_03-04_cescau4/2021-11-03-16-01-01.jpg"]
img_path_false = df_false["task_05-01-2021/2021-01-05-16-29-23.jpg"]
img_path_to_valid = df_valid["task_20210228/20210228-145102_(11.0).jpg"]
img_path_to_test = df_test["task_2021-03-01_09/20210301-090002_(13.0).jpg"]

#display(img_path_true)


a = 0#il y a bien des images qui sont dans les deux sets
for img_path in img_path_true:
    if img_path in img_path_to_valid.values:
        a+=1
print("a vaut:", a)
b = 0 #il y a bien des images qui sont dans le trainset et le validset
for img_path in img_path_false:
    if img_path in img_path_to_valid.values:
        b+=1

print("b vaut:", b)

c = 0
for img_path in img_path_to_test:
    if img_path in img_path_true.values:
        c+=1
print("c vaut:", c)

d = 0
for img_path in img_path_to_test:
    if img_path in img_path_false.values:
        d+=1
print("d vaut:", d)