from mss import mss
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import time
import json
import pyautogui


#top 28 je klasik
monitor = {"top": 313, "left": 0, "width": 800, "height": 30}
# monitor = {"top": 200, "left": 100, "width": 100, "height": 100}
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def main():
    distance = 1
    horizontalDict = {}

    with open('horizontal.json', 'r') as fp:
        horizontalDict = json.load(fp)


    if len(horizontalDict.keys()) != 0:
        distance = int(max(horizontalDict.keys(), key=lambda x:int(x)))


    time.sleep(5)
    while True:
        if distance > 800:
            break

        with mss() as sct:
            wasNotGood = False
            satisfying = False
            while not satisfying:
                beginning_frame = np.array(sct.grab(monitor))
                red_x = 500
                for i in range(800):
                    for j in range(30):
                        if beginning_frame[j][i][0] < 50 and beginning_frame[j][i][1] < 50 and beginning_frame[j][i][2] > 200:
                            red_x = i
                            break
                # if abs(400-red_x) > 10:
                #     wasNotGood = True
                if red_x < 399:
                    pyautogui.move(red_x - 400, 0)
                    print("recabr")
                    wasNotGood = True
                elif red_x > 401:
                    pyautogui.move(red_x - 400, 0)
                    print("recabr")
                    wasNotGood = True
                else:
                    satisfying = True

            if wasNotGood:
                distance -= 1

            pyautogui.move(distance, 0)
            second_frame = np.array(sct.grab(monitor))

            for i in range(30):
                    for j in range(800):
                        b = second_frame[i][j][0]
                        g = second_frame[i][j][1]
                        r = second_frame[i][j][2]
                        if not ( b < 50 and g < 50 and r > 150):
                            second_frame[i][j][0] = 0
                            second_frame[i][j][1] = 0
                            second_frame[i][j][2] = 0

            red_x = 500
            for i in range(800):
                for j in range(30):
                    if second_frame[j][i][2] > 150:
                        red_x = i
                        break
            print('Pixels traveled: ' + str(400 - red_x) +  " move was, " + str(distance))
            pyautogui.move(- 1 * distance, 0)


            distance += 1
            horizontalDict[distance] = 400 - red_x
            with open('horizontal.json', 'w') as fp:
                json.dump(horizontalDict, fp)




            time.sleep(0.2)

            cv2.imshow("result", second_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



def on_press(key):
    global shootActive
    if key == Key.home:
        shootActive = not shootActive
        print('ShootActive: ' + str(shootActive))

def addKeyboardListener():
    with Listener(
        on_press=on_press) as listener:
        listener.join()


if __name__ == '__main__':
    #_thread.start_new_thread(addKeyboardListener, ())
    main()

