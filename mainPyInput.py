from pynput.mouse import Button, Controller
import time

mouse = Controller()

while True:
    time.sleep(2)
    mouse.move(400, 0)
    mouse.click(Button.left, 1)
