import pyautogui
import time
from PIL import ImageGrab, Image
import os
from mss import mss
import cv2
import numpy

count = 0

#start = time.time()
#for i in range(30):
#    screenshot = pyautogui.screenshot(region=(0, 10, 800, 600))
#end = time.time()
#print(end - start)
#pyautogui, 30 screenshotova, 800x600 1-1.1 sekunda nezavisno od velicine i nezavisno od toga da li se cuvaju u fajl


#start = time.time()
#for i in range(30):
#    im2 = ImageGrab.grab(bbox =(0, 10, 800, 600)) 
#end = time.time()
#print(end - start)
#PIL Imagegrab, iste performanse kao i pyautogui


monitor = {"top": 150, "left": 150, "width": 500, "height": 500}
start = None
with mss() as sct:
    start = time.time()
    for i in range(30):
    #while True:
        img = numpy.array(sct.grab(monitor))
        cv2.imshow("OpenCV/Numpy normal", img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
end = time.time()
print(end - start)


#while True:
#    time.sleep(0.05)
#    screenshot = pyautogui.screenshot(imageFilename=str(count) + '.jpg', region=(0, 10, 800, 600))
#    count += 1