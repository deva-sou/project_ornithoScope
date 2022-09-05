from tqdm import tqdm

from tensorflow.keras.callbacks import Callback, TensorBoard

from .utils import (from_id_to_label_name,
                    compute_class_TP_FP_FN,
                    results_metrics_per_classes,
                    get_precision_recall_from_prediction_label,
                    get_precision_recall_from_prediction_box,
                    get_p_r_f1_global,
                    compute_bbox_TP_FP_FN,
                    BoundBox)


class MapEvaluation(Callback):
    """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
    """

    def __init__(self, yolo, generator,
                 iou_threshold=0.5,
                 score_threshold=0.5,
                 save_path=None,
                 period=1,
                 save_best=False,
                 save_name=None,
                 tensorboard=None,
                 label_names=[],
                 model_name=''):

        super().__init__()
        self._yolo = yolo
        self._generator = generator
        self._iou_threshold = iou_threshold
        self._score_threshold = score_threshold
        self._save_path = save_path
        self._period = period
        self._save_best = save_best
        self._save_name = save_name
        self._tensorboard = tensorboard
        self._label_names = label_names
        self._model_name = model_name

        self.bestMap = 0

        if not isinstance(self._tensorboard, TensorBoard) and self._tensorboard is not None:
            raise ValueError("Tensorboard object must be a instance from keras.callbacks.TensorBoard")


    def compute_P_R_F1(self):
        """
        Compute Precision, Recall and F1-Score.
        """

        # Lists TP, FP and FN per image as a list of dicts
        class_predictions, bbox_predictions = [], []

        # Lists predict boxes per image
        boxes_preds, bad_boxes_preds = {}, {}

        # Loop on every image of the test
        for i in tqdm(range(self._generator.size())):
            # Predict the image
            image, img_name = self._generator.load_image(i)
            pred_boxes = self._yolo.predict(image,
                                            iou_threshold=self._iou_threshold,
                                            score_threshold=self._score_threshold)

            #IL FAUT CHANGER LE PREDICT AFIN QU'IL PUISSE PREDIRE DES EMPTY AFIN DE POUVOIR CALCULER UN F1
            
            #print(f"{pred_boxes} ceci est la valeur de la pred box") #renvoie une liste vide si pas de bbox détectée
            
            # Load true results
            annotation_i = self._generator.load_annotation(i) #renvoie une liste contenant les annotations de l'image i sous forme des 4 coord + un élé
            
            
            #print(f"{annotation_i} ceci est la valeur de l'annotation")
            

            # Conver annotations to BoundBoxes
            if annotation_i != []:
                true_boxes = [
                    BoundBox(
                        box[0], box[1], box[2], box[3], 1,
                        [1 if c == box[4] else 0 for c in range(len(self._label_names))]
                    ) for box in annotation_i
                ]
                
            else:
                true_boxes = []
            
            #print(f"{true_boxes} ceci est la valeur de la true box")


            
            
            # Compute and add TP, FP and FN to the bbox prediction list
            bbox_preddicted = compute_bbox_TP_FP_FN(pred_boxes, true_boxes, self._label_names)
            bbox_predictions.append(bbox_preddicted)

            # Create class predicted dict
            class_preddicted = {} 
            class_preddicted['img_name'] = img_name
            class_preddicted['predictions_id'] = [box.get_label() for box in pred_boxes] #exctraction des id de la liste des boxes prédites
            class_preddicted['predictions_name'] = from_id_to_label_name(self._label_names, class_preddicted['predictions_id']) #renvoie le nom des oiseaux prédits 
            class_preddicted['score'] = [box.score for box in pred_boxes]
            if len(annotation_i[0]) == 0:
                class_preddicted['true_id'] = 0
                class_preddicted['true_name'] = ['EMPTY']
            else:
                class_preddicted['true_id'] = list(annotation_i[:,4])
                class_preddicted['true_name'] = from_id_to_label_name(self._label_names, list(annotation_i[:,4]))#renvoie le nom qui aurait dû être prédit
            
            
            # Compute and add TP, FP and FN to the class prediction list
            #A partir de cettel ligne on rajoute TP, FP et FN à class_preddicted

            print(class_preddicted)
            
            compute_class_TP_FP_FN(class_preddicted)  #on rajoute au dictionnaire class_preddicted les valeurs TP, FP et FN, par exemple TP=["VERREUR"]
            
            print(class_preddicted) #class_predicted est un dictionnaire qui contient les TP, FN et FP de l'image i, il contient aussi le nom des prédictions et tout cela pour chaque image
                                    
            class_predictions.append(class_preddicted)

            # Store predicted bounding box in 
            boxes_preds[img_name] = pred_boxes
            if (len(class_preddicted['FP'] + class_preddicted['FN'] + bbox_preddicted['FP'] + bbox_preddicted['FN']) > 0):
                bad_boxes_preds[img_name] = pred_boxes
        
        # Compute P, R and F1 with the class metrics
        class_metrics = get_precision_recall_from_prediction_label(class_predictions, self._label_names) #les labels_name sont ici les labels du backend (voir comment il appelle map-evaluate dans evaluate.py)
        class_res = results_metrics_per_classes(class_metrics)
        class_p_global, class_r_global, class_f1_global = get_p_r_f1_global(class_metrics)

        # Compute P, R and F1 with the bbox metrics
        bbox_metrics = get_precision_recall_from_prediction_box(bbox_predictions, self._label_names)
        bbox_res = results_metrics_per_classes(bbox_metrics)
        bbox_p_global, bbox_r_global, bbox_f1_global = get_p_r_f1_global(bbox_metrics)

        return (boxes_preds, bad_boxes_preds,
                class_predictions, class_metrics, class_res, class_p_global, class_r_global, class_f1_global,
                bbox_predictions, bbox_metrics, bbox_res, bbox_p_global, bbox_r_global, bbox_f1_global)
