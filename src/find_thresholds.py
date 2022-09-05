import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

from keras_yolov2.preprocessing import parse_annotation_csv
from keras_yolov2.utils import (bbox_iou,
                                from_id_to_label_name,
                                compute_class_TP_FP_FN,
                                get_p_r_f1_global,
                                get_precision_recall_from_prediction_label,
                                results_metrics_per_classes)


#d'abord évaluer un fichier config
#mettre ce fichier config en hardcode sur ce fichier, le lancer et obtenir le graphique

def load_k(k):
    """
    Loada the k-th image.
    """

    annots = []

    # Loop on all objects
    for obj in images[k]['object']:
        annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], config['model']['labels'].index(obj['name'])]
        annots += [annot]

    # If there is no objects, create empty result
    if len(annots) == 0:
        annots = [[]]

    return np.array(annots)


# Path to evaluation hisotry
pickle_path = "data/pickles/MobileNet_2022-08-08-19:53:49_0/boxes_MobileNet_input_test.p" #prendre le doc qui commence par boxes

# Open pickle
with open(pickle_path, 'rb') as fp:
    img_boxes = pickle.load(fp)

# Path to config filed use to evaluate
config_path = "config/pre_config/ADAM_OCS_v2_full_sampling_iNat_long.json"

# Open config file as a dict
with open(config_path) as config_buffer:
    config = json.load(config_buffer)

# Load test CSV input file
images = []
for valid_path in config['data']['test_csv_file']:
    images += parse_annotation_csv(valid_path,
                                    config['model']['labels'],
                                    config['data']['base_path'])[0]

# List labels
list_labels = config['model']['labels']

# Loop on
ious = np.linspace(0.0, 1.0, 10)
scores = np.linspace(0.0, 0.9, 10)

# Result in prf1 list
prf1 = []

it = 0
# Main loop
for iou_threshold in ious:
    res_iou = []
    for score_threshold in scores:
        predictions = []
        for k, img_name in enumerate(img_boxes):
            pred_boxes = img_boxes[img_name]

            # Select boxes with high scores
            pred_boxes = [box.copy() for box in pred_boxes if box.get_score() > score_threshold]

            # NMS
            for c in range(len(list_labels)):
                sorted_indices = list(reversed(np.argsort([box.classes[c] for box in pred_boxes])))

                for i in range(len(sorted_indices)):
                    index_i = sorted_indices[i]

                    if pred_boxes[index_i].classes[c] == 0:
                        continue
                    else:
                        for j in range(i + 1, len(sorted_indices)):
                            index_j = sorted_indices[j]

                            if bbox_iou(pred_boxes[index_i], pred_boxes[index_j]) >= iou_threshold:
                                pred_boxes[index_j].classes[c] = 0
            
            # Select boxes with high score now NMS is done
            pred_boxes = [box for box in pred_boxes if box.get_score() > score_threshold]
            
            # Extract boxes infos
            score = [box.score for box in pred_boxes]
            pred_labels = [box.get_label() for box in pred_boxes]

            # Store image infos
            labels_predicted = {}
            labels_predicted['img_name'] = img_name
            labels_predicted['predictions_id'] = pred_labels
            labels_predicted['predictions_name'] = from_id_to_label_name(list_labels, labels_predicted['predictions_id'])
            labels_predicted['score'] = score

            # Store expected values
            annotation_k = load_k(k)
            if len(annotation_k[0]) == 0:
                labels_predicted['true_id'] = 1
                labels_predicted['true_name'] = ['unknown']
            else:
                labels_predicted['true_id'] = list(annotation_k[:,4])
                labels_predicted['true_name'] = from_id_to_label_name(list_labels, list(annotation_k[:,4]))
            
            # Compute TP FP FN TN
            compute_class_TP_FP_FN(labels_predicted)
            predictions.append(labels_predicted)

        # Compute global results
        class_metrics = get_precision_recall_from_prediction_label(predictions, list_labels)
        class_res = results_metrics_per_classes(class_metrics)
        p_global, r_global, f1_global = get_p_r_f1_global(class_metrics)

        # Compute mean results
        P_list = []
        R_list = []
        F1_list = []
        for res in class_res:
            if res['Specie'] in list_labels:
                P_list.append(res['Precision'])
                R_list.append(res['Rappel'])
                F1_list.append(res['F-score'])
        p_mean, r_mean, f1_mean =  np.mean(P_list), np.mean(R_list), np.mean(F1_list)
        
        # Add new results
        res_iou.append([p_global, r_global, f1_global, p_mean, r_mean, f1_mean])

        # Show progress
        it += 1
        progress = round(it / len(ious) / len(scores) * 100)
        print(f' {progress: 3}%' , end='\r')

    prf1.append(res_iou)

titles = [
    'Global precision',
    'Global recall',
    'Global f1-score',
    'Mean precision',
    'Mean recall',
    'Mean f1-score'
]

zlabels = ['Precision', 'Recall', 'F1-Score']

# Create scores and ious meshes
scores_mesh, ious_mesh = np.meshgrid(scores, ious)

# P-R-F1 As array
prf1 = np.array(prf1)
for i in range(len(titles)):

    # Plot figures one by one
    plt.figure(titles[i])
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(scores_mesh, ious_mesh, prf1[:, :, i], rstride=1, cstride=1, cmap='twilight_shifted', edgecolor='none')
    ax.set_title(titles[i])
    ax.set_xlabel('Score threshold')
    ax.set_ylabel('IoU threshold')
    ax.set_zlabel(zlabels[i % 3])
    plt.colorbar(surf)

plt.show()
