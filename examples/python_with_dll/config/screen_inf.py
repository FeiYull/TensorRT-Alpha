import cv2
import numpy as np
import mss
import win32api

cap = mss.mss()
def grab_screen_mss(monitor):
    return cv2.cvtColor(np.array(cap.grab(monitor)), cv2.COLOR_BGRA2BGR)

def get_parameters():
        x, y = get_screen_size().values()
        return 0, 0, x, y

def get_screen_size():
    wide = win32api.GetSystemMetrics(0)
    high = win32api.GetSystemMetrics(1)
    return {"wide": wide, "high": high}