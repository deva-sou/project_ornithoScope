import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras
from .utils import compute_overlap, compute_ap, from_id_to_label_name, get_TP_FP_FN_TN, results_metrics_per_classes, get_precision_recall_from_prediction, print_results_metrics_per_classes, get_p_r_f1_global
from tensorflow.python.ops import summary_ops_v2
import pickle


class MapEvaluation(keras.callbacks.Callback):
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

        if not isinstance(self._tensorboard, keras.callbacks.TensorBoard) and self._tensorboard is not None:
            raise ValueError("Tensorboard object must be a instance from keras.callbacks.TensorBoard")

    def on_epoch_end(self, epoch, logs={}):

        if epoch % self._period == 0 and self._period != 0:
            precision,recall,f1score,_map, average_precisions = self.evaluate_map()
            print('\n')
            for label, average_precision in average_precisions.items():
                print(self._yolo.labels[label], '{:.4f}'.format(average_precision))
            print('mAP: {:.4f}'.format(_map))

            if self._save_best and self._save_name is not None and _map > self.bestMap:
                print("mAP improved from {} to {}, saving model to {}.".format(self.bestMap, _map, self._save_name))
                self.bestMap = _map
                self.model.save(self._save_name)
            else:
                print("mAP did not improve from {}.".format(self.bestMap))

            if self._tensorboard is not None:
                with summary_ops_v2.always_record_summaries():
                    with self._tensorboard._val_writer.as_default():
                        name = "mAP"  # Remove 'val_' prefix.
                        summary_ops_v2.scalar('epoch_' + name, _map, step=epoch)

    def evaluate_map(self):
        precisions,recalls,f1_scores,average_precisions = self._custom_p_r_f1_calculus()
        _map = sum(average_precisions.values()) / len(average_precisions)
        return precisions,recalls,f1_scores,_map, average_precisions

    def _custom_p_r_f1_calculus(self):
        # get labels predictions
        predictions = []
        list_labels = self._label_names
        for i in range(self._generator.size()): # generator size = number of tested images
            labels_predicted = {}
            raw_image, img_name = self._generator.load_image(i)
            raw_height, raw_width, _ = raw_image.shape  
            pred_boxes = self._yolo.predict(raw_image,
                                            iou_threshold=self._iou_threshold,
                                            score_threshold=self._score_threshold)
            score = [box.score for box in pred_boxes]
            pred_labels = [box.get_label() for box in pred_boxes]
            labels_predicted['img_name'] = img_name
            labels_predicted['predictions_id'] = pred_labels
            labels_predicted['predictions_name'] = from_id_to_label_name(list_labels,labels_predicted['predictions_id'])
            labels_predicted['score'] = score
            annotation_i = self._generator.load_annotation(i)
            if len(annotation_i[0]) == 0:
                labels_predicted['true_id'] = 1
                labels_predicted['true_name'] = ['unknown']
            else:
                labels_predicted['true_id'] = list(annotation_i[:,4])
                labels_predicted['true_name'] = from_id_to_label_name(list_labels,list(annotation_i[:,4]))
            get_TP_FP_FN_TN(labels_predicted)
            predictions.append(labels_predicted)
            #print(labels_predicted)
            if i%100 == 0:
                print("pred ",i,' done')
            
        #print(predictions)
                
        print('calculing done')
        print('\nFinal results:')
        pickle.dump(predictions, open( f"keras_yolov2/pickles/prediction_TP_FP_FN_{self._model_name}.p", "wb" ) )
        class_metrics = get_precision_recall_from_prediction(predictions, list_labels)
        pickle.dump(class_metrics, open( f"keras_yolov2/pickles/TP_FP_FN_{self._model_name}.p", "wb" ) )
        class_res = results_metrics_per_classes(class_metrics)
        average_precisions = 0
        # print(f'results\n, {precisions}\n {recalls}\n, {f1_scores}')
        print_results_metrics_per_classes(class_res)
        pickle.dump(class_res, open( f"keras_yolov2/pickles/P_R_F1_{self._model_name}.p", "wb" ) )
        p_global, r_global,f1_global = get_p_r_f1_global(class_metrics)
        print(f"Globals: P={p_global} R={r_global} F1={f1_global}")
        pickle.dump(class_res, open( f"keras_yolov2/pickles/P_R_F1_global_{self._model_name}.p", "wb" ) )        
        exit(0)
               
        
        return precisions,recalls,f1_scores,average_precisions

    
    def _calc_avg_precisions(self):
        # gather all detections and annotations
        # all_detections = [[None for _ in range(self._generator.num_classes())]
        #                   for _ in range(self._generator.size())]
        # all_annotations = [[None for _ in range(self._generator.num_classes())]
        #                    for _ in range(self._generator.size())]
        all_detections = [[[] for _ in range(self._generator.num_classes())]
                           for _ in range(self._generator.size())]
        all_annotations = [[[] for _ in range(self._generator.num_classes())]
                           for _ in range(self._generator.size())]
        for i in range(self._generator.size()): # generator size = number of tested images
            raw_image,img_name = self._generator.load_image(i)
            raw_height, raw_width, _ = raw_image.shape  

            # make the boxes and the labels
            # if i % 50 == 0 : 
            #     print(f"prediction number {i} done")
            print(f"\n \n \n prediction number {i}")
            pred_boxes = self._yolo.predict(raw_image,
                                            iou_threshold=self._iou_threshold,
                                            score_threshold=self._score_threshold)
            score = np.array([box.score for box in pred_boxes])
            if len(score) != 0:
                print('score ', score)
            pred_labels = np.array([box.get_label() for box in pred_boxes])
            if len(pred_labels) != 0:
                print('pred label ', pred_labels)
            if len(pred_boxes) > 0:
                print(pred_boxes)
                pred_boxes = np.array([[box.xmin * raw_width, box.ymin * raw_height, box.xmax * raw_width,
                                        box.ymax * raw_height, box.score] for box in pred_boxes])
                print('pred boxes ',pred_boxes)
            else:
                pred_boxes = np.array([[]])

            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(self._generator.num_classes()):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = self._generator.load_annotation(i)
            
            if annotations.shape[1] > 0:
                # copy ground truth to all_annotations
                for label in range(self._generator.num_classes()):
                    all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()  
                          
        print('all_detections \n', all_detections)
        # print('\n all_annotations ', all_annotations)
        # for anot in all_annotations:
        #     print('\n anot > \n')
        #     for a in anot:
        #         print('\n', a, '\n')
        # compute mAP by comparing all detections and all annotations
        average_precisions = {}
        precisions = {}
        recalls = {}
        f1_scores = {}
        #exit(0)
        for label in range(self._generator.num_classes()):
            print("Calculation on label: ", label)
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(self._generator.size()):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += len(annotations)
                detected_annotations = []
                if len(detections) != 0: 
                    print(f"detections {detections} \n label {label}")
                    print(f"annotations {annotations}")


                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= self._iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
            f1_score = 2*precision * recall/(precision + recall)
            print(f"label {label}, precision {precision}, recall {recall}, f1_score {f1_score}")
            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[label] = average_precision
            precisions[label] = precision
            recalls[label] = recall
            f1_scores[label] = f1_score

        print('calculing done')
        print('\n \n Final results')
        print('precision', precision)
        print(' recall', recall)
        print('f1_scores', f1_scores)
        print('average_precisions', average_precisions) 
        print('\n end of p,r,f1 score calculus \n')       
        return precisions,recalls,f1_scores,average_precisions
