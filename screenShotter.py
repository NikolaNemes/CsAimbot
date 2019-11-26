import numpy
from mss import mss
from pynput.keyboard import Key, Listener
import cv2
from scipy.misc import imsave, imread
from os import listdir
import math

path = '../ImagesCloseup/'
files = [int(''.join(filter(str.isdigit, f))) for f in listdir(path)]
monitor = {"top": 270, "left": 340, "width": 120, "height": 120}
img = numpy.empty(3)
saveReady = False
prefix = 'pos'
sct = mss()
img_num = max(files) + 1
def on_press(key):
    global saveReady
    global prefix
    global sct
    global img
    global img_num
    if key == Key.esc:
        return False
    if not hasattr(key, 'char'):
        return
    if key.char == 'p':
        saveReady = True
        prefix = 'pos'
        img = numpy.array(sct.grab(monitor))
        cv2.imshow("OpenCV/Numpy normal", img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
    elif key.char == 'o':
        saveReady = True
        prefix = 'neg'
        img = numpy.array(sct.grab(monitor))
        cv2.imshow("OpenCV/Numpy normal", img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
    elif key.char == 'l':
        if saveReady:
            saveReady = False
            imsave(path + prefix + str(img_num) + '.png', img)
            img_num += 1
with Listener(
        on_press=on_press) as listener:
    listener.join()