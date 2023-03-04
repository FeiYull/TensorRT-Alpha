from ctypes import *
from threading import Thread
import cv2
import numpy as np
import numpy.ctypeslib as npct
from pygame.time import Clock
from config.screen_inf import get_parameters, grab_screen_mss


class Detector:
    def __init__(
            self, dll_path, trt_path, window_width=640, window_height=640, conf_thresh=0.25, iou_thresh=0.45,
            num_class=80):
        self.yolo = CDLL(dll_path)
        self.max_bbox = 50

        self.yolo.Detect.argtypes = [c_void_p, c_int, c_int, POINTER(c_ubyte),
                                     npct.ndpointer(dtype=np.float32, ndim=2, shape=(self.max_bbox, 6),
                                                    flags="C_CONTIGUOUS")]

        self.yolo.Init.argtypes = [c_char_p, c_int, c_int, c_float, c_float, c_int]
        self.yolo.Init.restype = c_void_p

        self.c_point = self.yolo.Init(trt_path.encode('utf-8'), window_width, window_height, conf_thresh, iou_thresh,
                                      num_class)

    def predict(self, img):
        rows, cols = img.shape[0], img.shape[1]
        res_arr = np.zeros((self.max_bbox, 6), dtype=np.float32)
        self.yolo.Detect(self.c_point, c_int(rows), c_int(cols), img.ctypes.data_as(POINTER(c_ubyte)), res_arr)
        self.bbox_array = res_arr[~(res_arr == 0).all(1)]
        return self.bbox_array


class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# 对屏幕指定的区域录屏，并推理
if __name__ == '__main__':

    def img_grab_thread():

        global frame
        global monitor
        clock = Clock()

        while True:
            frame = grab_screen_mss(monitor)
            clock.tick(200)


    def img_pred_thread():

        global frame
        global source_w
        global source_h
        det = Detector(dll_path="./python_dll.dll", trt_path="./yolov8n.trt", window_width=source_w,
                       window_height=source_h)
        clock = Clock()

        windows_title = "cvwindow"
        cv2.namedWindow(windows_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)

        max_w = 576
        max_h = 324
        if source_h > max_h or source_w > max_w:
            cv2.resizeWindow(windows_title, max_w, source_h * max_w // source_w)

        while True:
            aims = det.predict(frame)
            for aim in aims:
                cv2.rectangle(frame, (int(aim[0]), int(aim[1])), (int(aim[2]), int(aim[3])), (0, 255, 0), 2)
                det_info = class_names[int(aim[4])] + " " + str(aim[5])
                cv2.putText(frame, det_info, (int(aim[0]), int(aim[1])), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 255), 1,
                            cv2.LINE_AA)

            cv2.putText(frame, "FPS:{:.1f}".format(clock.get_fps()), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 235), 4)
            cv2.imshow('cvwindow', frame)
            cv2.waitKey(1)

            clock.tick(200)


    # 4:3 800x600 center region detect
    source_w = int(800)
    source_h = int(600)

    _, _, x, y = get_parameters()
    top_x = (x // 2) - (source_w // 2)
    top_y = (y // 2) - (source_h // 2)

    monitor = {'left': top_x, 'top': top_y, 'width': source_w, 'height': source_h}

    frame = None

    # To demonstrate the inference speed more intuitively,
    # two threads are used here:
    # img_grab_thread for image fetching
    # img_pred_thread for inference
    # Lock is not used here, so the display effect may be poor if the image fetching speed is too high
    Thread(target=img_grab_thread).start()
    Thread(target=img_pred_thread).start()

# VideoCapture predict demo
if __name__ == '__main__OFF':
    cap = cv2.VideoCapture('./people.mp4')

    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    det = Detector(dll_path="./yoloDemo.dll", trt_path="./yolov8n.trt", window_width=source_w, window_height=source_h)

    clock = Clock()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        aims = det.predict(frame)

        # do something here
        for aim in aims:
            cv2.rectangle(frame, (int(aim[0]), int(aim[1])), (int(aim[2]), int(aim[3])), (0, 255, 0), 2)
            det_info = class_names[int(aim[4])] + " " + str(aim[5])
            cv2.putText(frame, det_info, (int(aim[0]), int(aim[1])), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 255), 1,
                        cv2.LINE_AA)

        cv2.imshow('cvwindow', frame)
        cv2.waitKey(1)

        print('pred fps: ', clock.get_fps())
        clock.tick(5)

    cap.release()
    cv2.destroyAllWindows()
