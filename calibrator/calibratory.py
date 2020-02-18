from mss import mss
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import time
import json
import pyautogui


#top 28 je klasik
monitor = {"top": 28, "left": 385, "width": 30, "height": 600}
# monitor = {"top": 200, "left": 100, "width": 100, "height": 100}
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def main():

    distance = 1
    verticalDict = {}

    with open('vertical.json', 'r') as fp:
        verticalDict = json.load(fp)


    if len(verticalDict.keys()) != 0:
        distance = int(max(verticalDict.keys(), key=lambda x:int(x)))


    time.sleep(5)
    while True:
        if distance > 400:
            break

        with mss() as sct:
            wasNotGood = False
            satisfying = False
            while not satisfying:
                beginning_frame = np.array(sct.grab(monitor))
                red_y = 100
                done = False
                for i in range(600):
                    for j in range(30):
                        if beginning_frame[i][j][0] < 50 and beginning_frame[i][j][1] < 50 and beginning_frame[i][j][2] > 200:
                            red_y = i
                            done = True
                            break
                    if done: 
                        break

                # if abs(400-red_y) > 10:
                #     wasNotGood = True
                if red_y < 297:
                    pyautogui.move(0, red_y - 300)
                    print("recabr")
                    wasNotGood = True
                elif red_y > 303:
                    pyautogui.move(0, red_y - 300)
                    print("recabr")
                    wasNotGood = True
                else:
                    satisfying = True

            if wasNotGood:
                distance -= 1

            pyautogui.move(0, -distance)
            second_frame = np.array(sct.grab(monitor))

            for i in range(600):
                for j in range(30):
                    b = second_frame[i][j][0]
                    g = second_frame[i][j][1]
                    r = second_frame[i][j][2]
                    if not ( b < 50 and g < 50 and r > 150):
                        second_frame[i][j][0] = 0
                        second_frame[i][j][1] = 0
                        second_frame[i][j][2] = 0

            red_y = 500
            done = False
            for i in range(600):
                for j in range(30):
                    if second_frame[i][j][2] > 150:
                        red_y = i
                        done = True
                        break
                if (done):
                    break
            print('Pixels traveled: ' + str(red_y - 300) +  " move was, " + str(distance))
            pyautogui.move(0, distance)


            distance += 1
            verticalDict[distance] = red_y - 300
            with open('vertical.json', 'w') as fp:
                json.dump(verticalDict, fp)




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

