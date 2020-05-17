#-*- coding: utf-8 -*-

import os
import time
import threading
import RPi.GPIO as GPIO
from copy import deepcopy

left_enable = 16
left_motor_positiv = 20
left_motor_negativ = 21
right_enable = 13
right_motor_positiv = 19
right_motor_negativ = 26

# 可改
speed = 100
                                                                                                
def init_gpio():
    GPIO.setmode(GPIO.BCM)
    # 这三行不重要，只是用来清空GPIO的设置的，因为上次程序执行完可能没有正常的退出。
    GPIO.setup(22, GPIO.OUT)
    GPIO.cleanup()

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(left_motor_positiv, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(left_motor_negativ, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(right_motor_positiv, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(right_motor_negativ, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(left_enable, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(right_enable, GPIO.OUT, initial=GPIO.HIGH)

def carMoveForward():
    GPIO.output(left_motor_positiv, GPIO.HIGH)
    GPIO.output(left_motor_negativ, GPIO.LOW)
    GPIO.output(right_motor_positiv, GPIO.HIGH)
    GPIO.output(right_motor_negativ, GPIO.LOW)
    leftMotorPwm.ChangeDutyCycle(speed-2.5)
    rightMotorPwm.ChangeDutyCycle(speed)

def carMoveBackward():
    GPIO.output(left_motor_positiv, GPIO.LOW)
    GPIO.output(left_motor_negativ, GPIO.HIGH)
    GPIO.output(right_motor_positiv, GPIO.LOW)
    GPIO.output(right_motor_negativ, GPIO.HIGH)
    leftMotorPwm.ChangeDutyCycle(speed-2)
    rightMotorPwm.ChangeDutyCycle(speed)

def carMoveForwardAndTurnLeft():
    GPIO.output(left_motor_positiv, GPIO.HIGH)
    GPIO.output(left_motor_negativ, GPIO.LOW)
    GPIO.output(right_motor_positiv, GPIO.HIGH)
    GPIO.output(right_motor_negativ, GPIO.LOW)
    leftMotorPwm.ChangeDutyCycle(5)
    rightMotorPwm.ChangeDutyCycle(100)
    
def carMoveForwardAndTurnRight():
    GPIO.output(left_motor_positiv, GPIO.HIGH)
    GPIO.output(left_motor_negativ, GPIO.LOW)
    GPIO.output(right_motor_positiv, GPIO.HIGH)
    GPIO.output(right_motor_negativ, GPIO.LOW)
    leftMotorPwm.ChangeDutyCycle(100)
    rightMotorPwm.ChangeDutyCycle(5)
    
def carMoveBackwardAndTurnLeft():
    GPIO.output(left_motor_positiv, GPIO.LOW)
    GPIO.output(left_motor_negativ, GPIO.HIGH)
    GPIO.output(right_motor_positiv, GPIO.LOW)
    GPIO.output(right_motor_negativ, GPIO.HIGH)
    leftMotorPwm.ChangeDutyCycle(5)
    rightMotorPwm.ChangeDutyCycle(35)
    
def carMoveBackwardAndTurnRight():
    GPIO.output(left_motor_positiv, GPIO.LOW)
    GPIO.output(left_motor_negativ, GPIO.HIGH)
    GPIO.output(right_motor_positiv, GPIO.LOW)
    GPIO.output(right_motor_negativ, GPIO.HIGH)
    leftMotorPwm.ChangeDutyCycle(35)
    rightMotorPwm.ChangeDutyCycle(5)
    
def PwmStop():
    global leftMotorPwm, rightMotorPwm
    leftMotorPwm.stop()
    rightMotorPwm.stop()
    
def PwmInit():
    global leftMotorPwm, rightMotorPwm
    # 设置pwm引脚和频率为2000hz
    leftMotorPwm = GPIO.PWM(left_enable, 1000)
    leftMotorPwm.start(0)
    rightMotorPwm = GPIO.PWM(right_enable, 1000)
    rightMotorPwm.start(0)

def key_detect(direction):
    # 任何对键盘，鼠标，触摸屏等的外部操作都会加入events事件队列，在每次循环中会对目前为止加入事件队列的
    # 事件进行遍历，而我们需要关注的事件只有按下上下左右键盘和松开上开上下左右键盘的操作，所以每次循环只
    # 需要对这几个事件进行判断操作。而被遍历到的一个事件只会对应众多情况中的一种情况，且都是独立的，所以判断时
    # 各自用一个if，无法列出所有的，所以不能用if，else。值得一提的是，每次按下键盘，会自动携带一个松开的事件
    # 紧接在其后加入事件队列，所以用break来离开for循环以防止后续的误判。另外，按下或是松开键盘（无论新按下或新松开几个按键）
    #才会激活函数pygame.key.get_pressed()，才会获得一定输出，该输出是键盘当下按下的状态，以此来判断上下左右是否处于按下
    # 还是松开的状态。
    import pygame
    pygame.init()
    pygame.display.set_mode((300,300))

    global key_detect_over_flag
    key_detect_over_flag = False
    
    while True:
        events = pygame.event.get()
           
        for event in events:
            if event.type == pygame.KEYUP:
                key_input = pygame.key.get_pressed()
                if key_input[pygame.K_UP] == 0:
                    direction[0] = 0
                if key_input[pygame.K_DOWN] == 0:
                    direction[1] = 0
                if key_input[pygame.K_LEFT] == 0:
                    direction[2] = 0
                if key_input[pygame.K_RIGHT] == 0:
                    direction[3] = 0
                    
            
            if event.type == pygame.KEYDOWN:
                key_input = pygame.key.get_pressed()
                if key_input[pygame.K_ESCAPE] == 1:
                    over_flag = True
                    break
                if key_input[pygame.K_UP] == 1:
                    direction[0] = 1
                if key_input[pygame.K_DOWN] == 1:
                    direction[1] = 1
                if key_input[pygame.K_LEFT] == 1:
                    direction[2] = 1
                if key_input[pygame.K_RIGHT] == 1:
                    direction[3] = 1
                break
        if key_detect_over_flag == True:
            break
            
    pygame.display.quit()
    pygame.quit()

def key_control_car(direction, capture_started, capture_start_flag, is_capture_running, key):
    """
    只对最多两个按键进行回应。
    有六种按键组合可以触发小车的运动：上 1000，下0100，上左1010，上右1001，下左0110，下右0101。
    在while循环里判断direction的状态。
    try,except keyboard interupt 来cleangpio，backpwmstop
    """
    olddirection = deepcopy(direction)  # 用等于号的话相当于指针传递，因为列表也是对象，deepcopy适合于对象的拷贝赋值，这样新变量就是独立于就变量的变量了

    global key_detect_over_flag
    
    init_gpio()
    PwmInit()
    try:
        while True:
            if not sameDirection(olddirection, direction):
                olddirection = deepcopy(direction)
                if direction == [1,0,0,0]:
                    capture_start_flag[0] = True
                    print ("up")
                    carMoveForward()
                    key[0] = 1
                    
                elif direction == [0,1,0,0]:
                    print ("down")
                    carMoveBackward()
                    
                elif direction == [0,0,1,0]:
                    print ("left")
                    carMoveForwardAndTurnLeft()
                    key[0] = 2
                                        
                elif direction == [0,0,0,1]:
                    print("right")
                    carMoveForwardAndTurnRight() 
                    key[0] = 3   
                    
                elif direction == [0,1,0,1]:
                    print("back and right")
                    carMoveBackwardAndTurnRight()
                    
                elif direction == [0,1,1,0]:
                    print("back and left")
                    carMoveBackwardAndTurnLeft()

                elif direction == [1,1,0,0]:
                    print("try to stop recording")
                    key[0] = 5

                else:
                    print('stop')
                    PwmStop()
                    PwmInit()
                    key[0] = 4

            time.sleep(.001)  # 保证电机有一定的接收信号的时间
            
            # 只有当拍摄开始之后才会检查拍摄是否结束以结束小车的控制
            if capture_started[0] and not is_capture_running[0]:
                break
        
    finally:
        key_detect_over_flag = True
        PwmStop()
        GPIO.cleanup()
        
# same move status
def sameDirection(olddirection, newdirection):
    for i in range(len(olddirection)):
        if olddirection[i] != newdirection[i]:
            return False

    return True

# 
# if __name__ == "__main__":
#     over_flag = False
#     direction = [0,0,0,0]
#     get_key_in_thread=threading.Thread(target=key_detect)
#     #  when daemon is set to True, the thread is terminated when the main thread ends.
#     get_key_in_thread.setDaemon(True)
#     get_key_in_thread.start()
#     
#     key_control_car(direction)

