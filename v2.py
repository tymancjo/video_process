import cv2
import random
import copy

def play_videoFile(filePath):

    cap = cv2.VideoCapture(filePath)
    cv2.namedWindow('Video output window',cv2.WINDOW_AUTOSIZE)

    # making a matrix buffer for the video
    vid_buffer = []

    # reading initial frames to buffer
    # for _ in range(200):
    #     ret_val, frame = cap.read()
    #     vid_buffer.append(frame)

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    # fontScale
    fontScale = 1
    color = (255, 255, 255)
    # Line thickness of 2 px
    thickness = 1
       


    frm = 0    
    total_frame = 0
    prev_frm = -1
    frm_step = 1
    mark = ["[ ]","[>]","[<]"] 

    while True:

        if frm_step > 0 and frm == len(vid_buffer):
            ret_val, frame = cap.read()
            vid_buffer.append(frame)
            total_frame += frm_step 

            buffer_len = len(vid_buffer)

            if buffer_len > 100:
                del vid_buffer[0]

        buffer_len = len(vid_buffer)
        if frm >= buffer_len:
            frm = buffer_len-1

        display_frame = copy.copy(vid_buffer[frm])

        # Using cv2.putText() method
        txt_string = f"{mark[frm_step]} AbsFrame: {total_frame+frm-buffer_len+1}  BufferFrm: {frm} HeadPos: {total_frame}"

        prev_frm = frm
        frm += frm_step
        


        if frm < 0:
            frm = 0
            frm_step = 0

        # for y in range(10):
           #  for x in range(10):
              #   line_thickness = random.randint(1,5)
              #   x1 = 10+x*192
              #   x2 = x1+192
              #   y1 = y*100+random.randint(1,100)
              #   y2 = y*100+random.randint(1,100)

              #   cv2.line(display_frame, (x1, y1), (x2, y2), (random.randint(50,255), random.randint(50,255), random.randint(50,255)), thickness=line_thickness)

        
        image = cv2.putText(display_frame, txt_string, org, font, 
                           fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Video Life2Coding', display_frame)

        the_pressed_key = cv2.waitKey(1)
        if  the_pressed_key == 27:
            break  # esc to quit

        elif the_pressed_key == ord('j'):
            frm_step = -1

        elif the_pressed_key == ord('k'):
            frm_step = 0

        elif the_pressed_key == ord('l'):
            frm_step = 1

    cv2.destroyAllWindows()

def main():
    play_videoFile('/Users/tymancjo/LocalGit/video/sc_data_example/11435.mp4')

if __name__ == '__main__':
    main()