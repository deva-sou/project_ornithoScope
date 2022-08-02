from keras_yolov2.preprocessing import parse_annotation_xml, parse_annotation_csv
from keras_yolov2.utils import import_feature_extractor
import json
import csv

#Rappel du principe de la fonction main: le if main à la fin organisera l'exécution du script de la manière suivante: main sera exécuter puis quand main fera appel à une fonction il l'exécutera aussi entre temps


#but: on rempli d'abord le fichier csv avec les classes les moins représentées mais comme des fois il y a des espèces très représentées sur ces photos, une fois qu'on aura pris les 150 photos d'une classe on pourra par exemple avoir déjà 50 photo d'écurou. Il faudra donc d'adapter et compléter jusqu'à 300 avec des images d'écurou


#fonction pour classer les espèces par leur nombre d'appararitions

especes=["MESCHA", "SITTOR", "MESBLE", "MESNON", "PINARB", "ACCMOU", "ROUGOR", "VEREUR", "MOIDOM", "TOUTUR", "ECUROU", "PIEBAV", "MULGRI", "CAMPAG", "MESNOI", "MESHUP"]
#train_imgs est un dico qui comprends 4 clés (object, filename, width et height). Le chemin de l'image est contenu dans filename
train_imgs, _ = parse_annotation_csv("data/inputs/input_train.csv",
                                        ["MESCHA", "SITTOR", "MESBLE", "MESNON", "PINARB", "ACCMOU", "ROUGOR", "VEREUR", "MOIDOM", "TOUTUR", "ECUROU", "PIEBAV", "MULGRI", "CAMPAG", "MESNOI", "MESHUP"],
                                        "/home/acarlier/code/data_ornithoscope/p0133_bird_data/raw_data/")

def compteur():
    
    
    nbr_images_par_espece={}
    for mot in especes:
        nbr_images_par_espece[mot]=0 #dictionnaire qui contient le nom des espèces et le nombre d'images prises pour chacune

    for id in range(len(train_imgs)):
        train_img = train_imgs[id]
        bbox = train_img['object']
        #print(train_imgs[1])
        #print(len(bbox))
        
        for i in range(len(bbox)):
            #print(bbox[i]['name'])
            nbr_images_par_espece[bbox[i]['name']] += 1
                
    #print(nbr_images_par_espece)

    return nbr_images_par_espece


#ici fonction qui complète écrit sur un csv les images avec des espèces rares et les espèces communes ou non uqi s'y trouve aussi + renvoie dic récapitulatif du remplissage
def remplissage_caté_peu_représentées():
    
    #si le mot est inf à 300 alors on récupère tous les filename où on peut trouver mot dans object
    f = open('data/data_lucien_caped300.csv', 'w', encoding='UTF8') 
    writer = csv.writer(f)

    dic={}
    for mot in especes:
        dic[mot] = 0

    for mot in compteur():
        
        if compteur()[mot] < 300:  #300
            
            for id in range(len(train_imgs)):
                train_img = train_imgs[id]
                bbox = train_img['object']
                for i in range(len(bbox)):
                    xmin = bbox[i]['xmin']
                    xmax = bbox[i]['xmax']
                    ymin = bbox[i]['ymin']
                    ymax = bbox[i]['ymax']

                w = train_img['width']
                h = train_img['height']
                name = train_img['filename']
                
                if len(bbox) == 1:
                    if mot == bbox[0]['name']:
                        # write the data
                        writer.writerow([name, xmin, xmax, ymin, ymax, w, h])
                        dic[mot] += 1
                
                if len(bbox) > 1:
                    L = []
                    
                    for i in range(len(bbox)):
                        L.append(bbox[i]['name'])
                        
                    if mot in L:
                        writer.writerow([name, xmin, xmax, ymin, ymax, w, h])
                        
                        for palabras in L:
                            dic[palabras] += 1
    print(dic)
    print(compteur())
    return dic  


def main():

    return remplissage_caté_peu_représentées()





if __name__ == '__main__':
    main()