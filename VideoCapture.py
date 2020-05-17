#-*- coding: utf-8 -*-

import os
import io
os.environ['SDL_VIDEODRIVE'] = 'x11'
import time
import picamera
import picamera.array

# 可改
RESOURCE = '8'
TEST_CAPTURE = False     # 是否先删除已有的照片， False就是添加。

RECORDING_TIME_SECONDS = 2
FRAMERATE = 30
RESOLUTION_LEVEL = 1/2
WIDTH, HEIGHT = int(160*RESOLUTION_LEVEL), int(120*RESOLUTION_LEVEL)


class SplitFrames(object):
    
    def __init__(self, key):
        self._key = key
        self.frame_num = 0
        self.output = None
        self.stop_recording = False
        self.finish_time = None

    def write(self, buf):
        if self.stop_recording:
            return
        if buf.startswith(b'\xff\xd8'):
            if self._key[0] == 5:
                self.stop_recording = True
                self.finish_time = time.time()
                print("Recording stopped")
                return
            # Start of new frame; close the old one (if any) and
            # open a new output
            if self.output:
                self.output.close()
            self.frame_num += 1
            self.output = io.open('/home/pi/Desktop/images{}/{}_image_{}.jpg'.format(RESOURCE, time.time(), self._key[0]), 'wb')
        self.output.write(buf)


def capture(is_capture_running, capture_started, key):
    if not os.path.exists("/home/pi/Desktop/images"+RESOURCE):
        os.mkdir("/home/pi/Desktop/images"+RESOURCE)
    if TEST_CAPTURE:
        os.system("rm /home/pi/Desktop/images{}/*.jpg".format(RESOURCE))

    capture_started[0] = True
    is_capture_running[0] = True
    
    with picamera.PiCamera(resolution=(WIDTH, HEIGHT), framerate=FRAMERATE) as camera:
        # 根据摄像头实际情况判断是否要加这句上下翻转
        # camera.vflip = True
        # Give the camera some warm-up time
        time.sleep(2)
        output = SplitFrames(key)
        print("Start capture")
        start_time = time.time()
        camera.start_recording(output, format='mjpeg')
        camera.wait_recording(RECORDING_TIME_SECONDS)
        camera.stop_recording()
        if output.stop_recording:
            finish_time = output.finish_time
        else:
            finish_time = time.time()
        is_capture_running[0] = False

    print('Captured {} frames at {}fps in {}s'.format(output.frame_num, output.frame_num / (finish_time - start_time), finish_time - start_time))
    print("quit pi capture")


