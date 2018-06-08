import numpy as np
import cv2
import time
import mss
import mss.tools
from PIL import Image
import pyautogui


def control_game():
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)
    print('down')
    pyautogui.keyDown('w')
    time.sleep(3)
    print('up')
    pyautogui.keyUp('w')

def region_of_interest(img, verticles):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [verticles], 255)
    masked =cv2.bitwise_and(img, mask)
    return masked

def process_image(original_img):
    #processed_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    vertices = np.array([[10, 520], [10, 200], [200,200], [500,200], [640,300], [640,500]])
    processed_img = region_of_interest(original_img, vertices)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2= 300)
    return processed_img

def record_screen():
    with mss.mss() as sct:
        last_time = time.time()
        bbox=(0,20,640,520)
        while True:
        # for i in range(10):
            last_time = time.time()
            img = sct.grab(bbox)
            processed_img = process_image(np.array(img))
            cv2.imshow('Game', processed_img)
            #cv2.imshow('Game', np.array(img))
            # print('Time taken is {}'.format(time.time() - last_time))

            # Save it!
            #mss.tools.to_png(img.rgb, img.size, output='screenshot{}.png'.format(i))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            

if __name__ == '__main__':
    # control_game()
    record_screen()