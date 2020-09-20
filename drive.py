#-*- coding: utf-8 -*-

import os
import io
import glob
import time
import threading
import picamera
import picamera.array
from PIL import Image
import numpy as np

import RPi.GPIO as GPIO
import KeyControlCar
import tensorflow as tf
from keras.models import load_model
from tensorflow.python.keras.backend import set_session
#from tensorflow.python.keras.models import load_model

# kegai
RESOURCE = 8

GRAY_IMG = False
if GRAY_IMG:
    IMAGE_CHANNELS = 1
else:
    IMAGE_CHANNELS = 3
NORMALIZATION = True
RESOLUTION_LEVEL = 0.5
IMAGE_HEIGHT, IMAGE_WIDTH = int(120*RESOLUTION_LEVEL), int(160*RESOLUTION_LEVEL)
FRAMERATE = 30

print(IMAGE_HEIGHT, IMAGE_WIDTH)

# def get_max_prob_num(predictions_array):
#   """to get the integer of predition, instead of digit number"""
    
#   prediction_edit = np.zeros([1,5])
#   for i in range(0,5):
#       if predictions_array[0][i] == predictions_array.max():
#           prediction_edit[0][i] = 1
#           return i
#   return 2


latest_action_num = -1
def control_car(action_num):
    global latest_action_num
    if action_num != latest_action_num:
        latest_action_num = action_num

        if action_num == 0:
            print('Forward')
            KeyControlCar.carMoveForward()

        elif action_num== 1:
            print("Left")
            KeyControlCar.carMoveForwardAndTurnLeft()

        elif action_num == 2:
            print("Right")
            KeyControlCar.carMoveForwardAndTurnRight() 

        time.sleep(.001)  # 保证电机有一定的接收信号的时间


class ImageProcessor(threading.Thread):
    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.start()

    def run(self):
        global latest_time, model, graph, sess
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    self.stream.seek(0)
                    # Read the image and do some processing on it
                    image = Image.open(self.stream)
                    if GRAY_IMG:
                        image = image.convert("L") 
                    image_np = np.array(image).astype('float32')
                    if NORMALIZATION:
                        image_np /=  255
                    camera_data_array = image_np.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
                
                    # camera_data_array = np.expand_dims(image_np,axis = 0)

                    current_time = time.time()
                    if current_time>latest_time:
                        if current_time-latest_time>1:
                            print("*" * 30)
                            print(current_time-latest_time)
                            print("*" * 30)
                        latest_time = current_time

                        with graph.as_default():
                            set_session(sess)
                            predictions_array = model.predict(camera_data_array)
                        # print(predictions_array)
                        # action_num = get_max_prob_num(predictions_array)
                        action_num = predictions_array.argmax()
                        #print "action_num: {}".format(action_num)
# action_num = get_max_prob_num(predictions_array)
                        action_num = predictions_array.argmax()
                        #print "action_num: {}".format(action_num)

                        control_car(action_num)
                        control_car(action_num)
                        # Uncomment this line if you want to save images with prediction as name
                        # Warning: This will cause latency sometimes.
                        # image.save('%s_image%s.jpg' % (action_num,time.time()))
                except Exception as error:
                    print(error)
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the available pool
                    with self.owner.lock:
                        self.owner.pool.append(self)


class ProcessOutput(object):
    def __init__(self):
        self.done = False
        # Construct a pool of 4 image processors along with a lock
        # to control access between threads
        self.lock = threading.Lock()
        self.pool = [ImageProcessor(self) for i in range(4)]
        self.processor = None

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame; set the current processor going and grab
            # a spare one
            if self.processor:
                # handle the last frame
                self.processor.event.set()
            with self.lock:
                if self.pool:
                    # take out a threading from the threading pool to handle the frame that is got at this round.
                    # the reason of using more threading is maybe before next frame comes the last frame is not handled (for prediction and car action) finished yet
                    self.processor = self.pool.pop()
                else:
                    # No processor's available, we'll have to skip
                    # this frame; you may want to print a warning
                    # here to see whether you hit this case
                    print("No processor's available! This frame must be skipped.")
                    self.processor = None
        if self.processor:
            # store this frame and wait until next frame comes then this frame will be handled by this processor
            self.processor.stream.write(buf)

    def flush(self):
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion. First, add the current processor
        # back to the pool
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None
        # Now, empty the pool, joining each thread as we go
        while True:
            with self.lock:
                try:
                    proc = self.pool.pop()
                except IndexError:
                    pass # pool is empty
            proc.terminated = True
            proc.join()


def main():
    """get data, then predict the data, edited data, then control the car"""
    global model, graph, sess
    
    # model_loaded = glob.glob('model/*.h5')
    # for single_mod in model_loaded:
    #   model = load_model(single_mod)
    
    #tf_config = some_custom_config
    sess = tf.Session()
    #graph = tf.compat.v1.get_default_graph()
    graph = tf.get_default_graph()
    set_session(sess)
    print("loading model from source {}...".format(RESOURCE))
    model = load_model("model/car_model_from_source_{}.h5".format(RESOURCE))
    print("loaded model")
    
    
    try:
        KeyControlCar.init_gpio()
        KeyControlCar.PwmInit()
        with picamera.PiCamera(resolution=(IMAGE_WIDTH, IMAGE_HEIGHT)) as camera:
            # uncomment this line and the camera images will be upside down
            # camera.vflip = True
            time.sleep(2)
            output = ProcessOutput()
            camera.start_recording(output, format='mjpeg')
            while not output.done:
                camera.wait_recording(1)
            camera.stop_recording()
    finally:
        if not output.done:
            camera.stop_recording()
        KeyControlCar.PwmStop()
        GPIO.cleanup()
        print ("over")


if __name__ == '__main__':
    latest_time = time.time()
    main()
