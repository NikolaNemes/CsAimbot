import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import json
class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/counterstrikeclasses.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (608, 800),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self):
        self.__dict__.update(self._defaults) # set up default values
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            # boxed_image = image
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            # boxed_image = image
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        classIndex = 0
        labels = ['sass', 'leet']
        for i in range(len(out_boxes)): 
            box = out_boxes[i]
            score = out_scores[i]
            predicted_class = labels[out_classes[i]]
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            #My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                   outline=self.colors[classIndex])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[classIndex])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        # print(end - start)
        return image, out_boxes, out_classes, out_scores

    def close_session(self):
        self.sess.close()

#return a list of (image path, all image boxes, all boxes classes)
def decipher_ground_truth():
    with open('evaluationImages/annotations.txt') as f:
        lines = [line.rstrip() for line in f]
    retval = []
    for line in lines:
        line_split = line.split()
        image_path = line_split[0]
        boxes_string = line_split[1:]
        boxes = []
        classes = []
        for box in boxes_string:
            box_details = box.split(',')
            classes.append(int(box_details[4]))
            boxes.append(list(map(lambda x: int(x), box_details[0:4])))
        retval.append((image_path, boxes, classes))
    return retval


def calc_iou( gt_bbox, pred_bbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and 
    return the IoU ratio
    '''
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt= gt_bbox
    y_topleft_p, x_topleft_p, y_bottomright_p, x_bottomright_p= pred_bbox
    
    if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt> y_bottomright_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_topleft_p > x_bottomright_p) or (y_topleft_p> y_bottomright_p):
        raise AssertionError("Predicted Bounding Box is not correct",x_topleft_p, x_bottomright_p,y_topleft_p,y_bottomright_gt)
        
         
    #if the GT bbox and predcited BBox do not overlap then iou=0
    if(x_bottomright_gt< x_topleft_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
        
        return 0.0
    if(y_bottomright_gt< y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
        
        return 0.0
    if(x_topleft_gt> x_bottomright_p): # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
        
        return 0.0
    if(y_topleft_gt> y_bottomright_p): # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
        
        return 0.0
    
    
    GT_bbox_area = (x_bottomright_gt -  x_topleft_gt + 1) * (  y_bottomright_gt -y_topleft_gt + 1)
    Pred_bbox_area =(x_bottomright_p - x_topleft_p + 1 ) * ( y_bottomright_p -y_topleft_p + 1)
    
    x_top_left =np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])
    
    intersection_area = (x_bottom_right- x_top_left + 1) * (y_bottom_right-y_top_left  + 1)
    
    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
   
    return intersection_area/union_area

#image details - image path, list of annotated boxes, list of classes for those boxes
#out_boxes, out_classes, out_scores - yolo results for this image
#target class - 
#output - img-id -> truepos, falsepos, falseneg
def calculateResults(image_details_ground_truth, out_boxes, out_classes, out_scores, target_class):
    image_details_ground_truth_class = [image_details_ground_truth[0], [], []]
    for i in range(len(image_details_ground_truth[2])):
        if image_details_ground_truth[2][i] == target_class:
            image_details_ground_truth_class[1].append(image_details_ground_truth[1][i])
            image_details_ground_truth_class[2].append(image_details_ground_truth[2][i])

    results = {'true_pos': 0, 'false_pos': 0, 'false_neg': 0}
    found_ground_truth_boxes = [False for i in range(len(image_details_ground_truth_class[1]))]
    for i in range(0, len(out_boxes)):
        if out_classes[i] == target_class:
            out_box = out_boxes[i]
            all_ious = []
            for j in range(0, len(image_details_ground_truth_class[1])):
                if (found_ground_truth_boxes[j] == True):
                    all_ious.append(-1)
                    continue
                ground_truth_out_box = image_details_ground_truth_class[1][j]
                all_ious.append(calc_iou(ground_truth_out_box, out_box))
            if len(all_ious) > 0:
                max_iou = max(all_ious)
                if max_iou > 0.5:
                    results['true_pos'] = results['true_pos'] + 1
                    found_ground_truth_boxes[all_ious.index(max_iou)] = True
                else:
                    results['false_pos'] = results['false_neg'] + 1
    false_negs = 0
    for found in found_ground_truth_boxes:
        if not found:
            false_negs += 1
    results['false_neg'] = false_negs
    return results

def calc_precision_recall(image_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_positive=0
    false_positive=0
    false_negative=0
    for img_id, res in image_results.items():
        true_positive +=res['true_pos']
        false_positive += res['false_pos']
        false_negative += res['false_neg']
        try:
            precision = true_positive/(true_positive+ false_positive)
        except ZeroDivisionError:
            precision=0.0
        try:
            recall = true_positive/(true_positive + false_negative)
        except ZeroDivisionError:
            recall=0.0
    return (precision, recall)

def calc_precision_recall_both_classes(image_results_sass, image_results_leet):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_positive=0
    false_positive=0
    false_negative=0
    for img_id, res in image_results_sass.items():
        true_positive +=res['true_pos']
        false_positive += res['false_pos']
        false_negative += res['false_neg']
        try:
            precision = true_positive/(true_positive+ false_positive)
        except ZeroDivisionError:
            precision=0.0
        try:
            recall = true_positive/(true_positive + false_negative)
        except ZeroDivisionError:
            recall=0.0
    for img_id, res in image_results_leet.items():
        true_positive +=res['true_pos']
        false_positive += res['false_pos']
        false_negative += res['false_neg']
        try:
            precision = true_positive/(true_positive+ false_positive)
        except ZeroDivisionError:
            precision=0.0
        try:
            recall = true_positive/(true_positive + false_negative)
        except ZeroDivisionError:
            recall=0.0
    return (precision, recall)


def calc_f_score(precision_recall):
    precision = precision_recall[0]
    recall = precision_recall[1]
    return 2 * precision * recall / (precision + recall)

if __name__ == "__main__":
    yolo = YOLO()
    ground_truth = decipher_ground_truth()
    precision_recalls = []
    results_both_classes = []
    for target_class in (0, 1):
        results = {}
        for image_details in ground_truth:
            try:
                image = Image.open(image_details[0])
            except:
                print('cant open file ' + image_details[0])
                continue
            r_image, out_boxes, out_classes, out_scores = yolo.detect_image(image)
            results[image_details[0]] = calculateResults(image_details, out_boxes, out_classes, out_scores, target_class)
        results_both_classes.append(results)
        precision_recalls.append(calc_precision_recall(results))
    fscores = [calc_f_score(precision_recall) for precision_recall in precision_recalls]
    print('Macro average for F-score: ' + str(sum(fscores) / len(fscores)))
    print('Micro average for F-score: ' + str(calc_f_score(calc_precision_recall_both_classes(results_both_classes[0], results_both_classes[1]))))
    yolo.close_session()