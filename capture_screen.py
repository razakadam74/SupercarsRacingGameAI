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
    masked = cv2.bitwise_and(img, mask)
    return masked


def draw_line(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]),
                     (coords[2], coords[3]), [255, 255, 255], 3)
    except:
        pass


def process_image(original_img):
    processed_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    # vertices = np.array([[10, 520], [10, 100], [100, 200], [
    #                     500, 200], [620, 200], [620, 500]])

    height = original_img.shape[0]
    width = original_img.shape[1]

    #a3 = np.array( [[[10,10],[100,10],[100,100],[10,100]]], dtype=np.int32 )
    vertices = np.array( [[[0,height/2.5],[width,height/ 2.5],[width,height],[0,height]]], dtype=np.int32 )

    #vertices = np.array([[0, height],[width / 2, height / 2],[width, height]])
    processed_img = region_of_interest(processed_img, vertices)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    # edges
    # lines = cv2.HoughLinesP(processed_img, rho=6, theta=np.pi / 60,
    #                         threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25)
    lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 180,np.array([]),200, 15)

    # draw line in processed_image
    draw_line(processed_img, lines)
    return processed_img


def record_screen():
    with mss.mss() as sct:
        last_time = time.time()
        bbox = (0, 20, 640, 520)
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
