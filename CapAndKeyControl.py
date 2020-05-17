import time
import threading
import VideoCapture
import KeyControlCar

"""
先让小车可以运动，然后启动摄像头拍摄
按下q，结束拍摄，并且不再运动
"""

if __name__ == "__main__":
    # global variable
    capture_started = [False]
    direction = [0,0,0,0]
    capture_start_flag = [False]
    is_capture_running = [False]
    key = [0]
    
    # detecting key and controlling car are in two threads 
    key_detect_in_thread=threading.Thread(target=KeyControlCar.key_detect, args=(direction,))
    key_detect_in_thread.start()
    key_control_car_in_thread=threading.Thread(target=KeyControlCar.key_control_car, args=(direction, capture_started, capture_start_flag, is_capture_running,key,))
    key_control_car_in_thread.start()
    
    # video capture
    while not capture_start_flag[0]:
        time.sleep(.1)
        pass
    VideoCapture.capture(is_capture_running, capture_started, key)
    
    key_detect_in_thread.join()
    key_control_car_in_thread.join()
    print("Yeah, finished!!")
    
    