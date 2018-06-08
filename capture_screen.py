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

    try:
        l1,l2 = draw_lanes(original_img, lines)
        cv2.line(original_img, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
        cv2.line(original_img, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
    except Exception as e:
        print(str(e))

    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
            except Exception as e:
                print(str(e))
    except Exception as e:
        print(str(e))

    
    # draw line in processed_image
    draw_line(processed_img, lines)
    # processed_img = draw_lanes(processed_img, lines)
    return processed_img

def draw_lanes(img, lines, color=[0,255,255], thickness =3):
    try:
        ys = []
        for i in lines:
            for ii in i:
                ys += [ii[1], ii[3]]
        min_y = min(ys)
        max_y = img.shape[0]
        new_lines = []
        line_dict = {}

        for idx, i in enumerate(lines):
            for xyxy in i:
                x_coords = (xyxy[0], xyxy[2])
                y_coords = (xyxy[1], xyxy[3])
                A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                m,b = np.lstsq(A, y_coords)[0]

                x1= (min_y - b) / m
                x2 = (max_y - b) /m

                line_dict[idx] = [m,b, [int(x1), min_y, int(x2), max_y]]
                #new_lines.append([int(x1), min_y, int(x2),max_y])
        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]

            if len(final_lanes) == 0:
                final_lanes[m] = [[m,b,line]]
            else:
                found_copy =False

                for other_ms in final_lanes_copy:
                    if not found_copy:
                        if abs(other_ms * 1.2) > abs(m) > abs(other_ms * 0.9):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [[m,b,line]]
        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])
        
        top_lanes = sorted(line_counter.items(), key= lambda item: item[1])[::-1][:2]

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []

            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(np.mean(x1s)), int(np.mean(y1s)), int(np.mean(x2s)), int(np.mean(y2s)) 

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]
    except Exception as e:
        print(str(e))


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
