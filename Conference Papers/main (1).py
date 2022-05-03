import RPi.GPIO as GPIO
import time

PIO.setmode(GPIO.BOARD)
GPIO.setup(self.LedPin, GPIO.OUT)
self.pwm = GPIO.PWM(self.LedPin, 1000)
self.pwm.start(0)
for i in range(100):
    self.pwm.ChangeDutyCycle(i)
    time.sleep(0.1)
for i in range(100, 0, -1):
    self.pwm.ChangeDutyCycle(i)
    time.sleep(0.1)
pwm.stop()
GPIO.cleanup()