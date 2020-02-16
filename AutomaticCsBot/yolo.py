# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer
import time

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import math

from mss import mss
import pyautogui
from sklearn.svm import SVC # SVM klasifikator
import joblib
import _thread
from pynput.keyboard import Key, Listener

import json

monitor = {"top": 28, "left": 0, "width": 800, "height": 600}
svm_model = None
shotCounter = 0
verticalDict = {}
horizontalDict = {}
shootActive = False

with open('MouseDicts/horizontalCorrected.json', 'r') as fp: 
    horizontalDict = json.load(fp)

with open('MouseDicts/verticalCorrected.json', 'r') as fp: 
    verticalDict = json.load(fp)



def centerDistance(box):
    top, left, bottom, right = box
    center = ((left + right) // 2, (top + bottom) // 2)
    return math.sqrt((center[0] - 400)**2 + (center[1] - 300)**2)

def aim_at_box(box):
    global shotCounter
    top, left, bottom, right = box

    if top < 300 and bottom > 300 and left < 400 and right > 400:
        pyautogui.click()
        shotCounter += 1
        return

    center = ((left + right) // 2, top + (bottom - top) // 100 * 65)
    dx = int(center[0] - 400)
    dy = int(center[1] - 300)
    dxresult = 0
    dyresult = 0
    if dx >= 0:
        dxresult = horizontalDict[str(dx)]
    else:
        dxresult = -horizontalDict[str(-dx)]
    if dy >= 0:
        dyresult = verticalDict[str(dy)]
    else:
        dyresult = - verticalDict[str(-dy)]
    print(top, bottom)
    print(left, right)
    print("-------------")
    
    pyautogui.move(dxresult, dyresult)
    time.sleep(0.05)
    pyautogui.click()
    shotCounter += 1

def addKeyboardListener():
    with Listener(
        on_press=on_press) as listener:
        listener.join()

def on_press(key):
    global shootActive
    if key == Key.home:
        shootActive = not shootActive
        print('ShootActive: ' + str(shootActive))

def reshape_data(img_feature_set):
    nsamples = 1
    nx, ny, channels = img_feature_set.shape
    return img_feature_set.reshape((nsamples, nx*ny*channels))[0]


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
        start = timer()

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

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        classIndex = 0
        labels = ['sass', 'leet']

        terrorists_boxes = []
        if shootActive:
            for i in range(len(out_boxes)):
                if labels[out_classes[i]] == 'leet':
                    terrorists_boxes.append((out_boxes[i], out_classes[i]))

            if len(terrorists_boxes) != 0 and shotCounter < 3:
                box = max(terrorists_boxes, key=lambda x: x[1])[0]
                aim_at_box(box)

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
                    [left + i + (right - left)//2, top + i + (bottom - top) // 2, right - i - (right - left) // 2, bottom - i - (bottom - top) // 2], outline=self.colors[classIndex]
                )
                draw.rectangle(
                    [397, 297, 403, 303], outline=self.colors[classIndex]
                )
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[classIndex])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        # print(end - start)
        return image

    def close_session(self):
        self.sess.close()


def load_svm_classifier():
    return joblib.load('model_data/svm.joblib')

def detect_video(yolo):
    import cv2
    global svm_model
    svm_model = load_svm_classifier()
    _thread.start_new_thread(addKeyboardListener, ())
    accum_time = 0
    shot_accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    pyautogui.PAUSE = 0
    while True:
        with mss() as sct:
            frame = sct.grab(monitor)
            image = Image.frombytes("RGB", frame.size, frame.bgra, "raw", "BGRX")
            # frame = np.array(sct.grab(monitor))
            # image = Image.fromarray(frame)
            image = yolo.detect_image(image)
            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            shot_accum_time += exec_time
            curr_fps = curr_fps + 1
            if shot_accum_time > 0.5:
                global shotCounter
                shotCounter = 0
                shot_accum_time = 0
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps) + (' ' + 'SHOOTING: ON' if shootActive else 'SHOOTING: OFF')
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(24, 240, 28), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    yolo.close_session()

