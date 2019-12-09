import pyautogui
import time
from PIL import ImageGrab, Image
import os
from mss import mss
import cv2
from sklearn.svm import SVC # SVM klasifikator
import joblib
import numpy as np
from ahk import AHK
from pynput.keyboard import Key, Listener
import _thread
from win32 import win32gui
from pythonwin import win32ui
from win32.lib import win32con
from win32 import win32api

#globals
count = 0
ahk = AHK(executable_path="F:\\Program Files\\AutoHotkey\\AutoHotkey.exe")
width = 600
height = 800
top = 300
left = 370
hwin = win32gui.GetDesktopWindow()

shootActive = False

def load_model():
    if os.path.isfile('model.joblib'):
        return joblib.load('model.joblib')
    else:
        return SVC(kernel='linear', probability=True)


clf_svm = load_model()

nbins = 9 # broj binova
cell_size = (8, 8) # broj piksela po celiji
block_size = (3, 3) # broj celija po bloku

hog = cv2.HOGDescriptor(_winSize=(60 // cell_size[1] * cell_size[1], 
                                    60 // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)


def reshape_data(img_feature_set):
    input_data = np.vstack((img_feature_set))
    nsamples = 1
    nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))[0]


def extract_features(img, shape):
    return hog.compute(img)


def predict(img):
    shape = img.shape
    feature_set = extract_features(img, shape)
    feature_set = np.array(reshape_data(feature_set)).reshape(1, -1)
    result = clf_svm.predict_proba(feature_set)
    return result

def grab_screen():

    hwindc = win32gui.GetWindowDC(hwin)



    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def main():
    monitor = {"top": 300, "left": 370, "width": 60, "height": 60}
    with mss() as sct:
        #for i in range(30):
        while True:
            img = grab_screen()
            #img = np.array(sct.grab(monitor))
            cv2.imshow("OpenCV/Numpy normal", img)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if shootActive and predict(img)[0][1] > 0.65:
                ahk.click()
        
def on_press(key):
    global shootActive
    if key == Key.home:
        shootActive = not shootActive
        print('ShootActive: ' + str(shootActive))

def addKeyboardListener():
    with Listener(
        on_press=on_press) as listener:
        listener.join()

def performance_comparisson():
    with mss() as sct:
        monitor = {"top": 300, "left": 370, "width": 600, "height": 800}
        print('Sct grab: ')
        start = time.time()
        for i in range(30):
            img = np.array(sct.grab(monitor))
        end = time.time()
        print('Time: ' + str(end - start))
    print('PyWin32: ')
    start = time.time()
    for i in range(30):
        img = grab_screen()
    end = time.time()
    print('Time: ' + str(end - start))


if __name__ == '__main__':
    #_thread.start_new_thread(addKeyboardListener, ())
    #main()
    performance_comparisson()

