from ahk import AHK
import time

ahk = AHK(executable_path="F:\\Program Files\\AutoHotkey\\AutoHotkey.exe")

while True:
    ahk.mouse_move(100, 0, speed=1, relative=True)  # Moves the mouse reletave to the current position
    ahk.click()  # Click the primary mouse button
    time.sleep(2)