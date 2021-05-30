import cv2
import copy
import sys
import time
import csv
import numpy as np
import math

def play_videoFile(filePath, data, vid_sync, data_sync):

    cap = cv2.VideoCapture(filePath)

    if not cap.isOpened(): 
        print(f"Can't open file {filePath}")
        return

    v_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    v_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    v_fps    = cap.get(cv2.CAP_PROP_FPS)

    print(f"Got video: {v_width}x{v_height} px, len:{v_length}frames, fps:{v_fps}")
    
    cv2.namedWindow('Video output window',cv2.WINDOW_AUTOSIZE)

    # making a matrix buffer for the video
    vid_buffer = []

    # text related variables
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 0.7
    color = (255, 255, 255)
    thickness = 1
       


    buffer_mem = 500 #[MB]
    buffer_size = 100 #[frames]
    frm = 0    
    total_frame = 0
    prev_frm = -1
    frm_step = 0
    mark = ["[-]","[>]","[<]"] 
    t_start = 0
    t_end = 1
    step = False
    first = True

    # Figuring out the max possible play frame of video or data
    n_samples, n_cols = data.shape
    # Need to analyze here the data sets and figure out the first video and data frame that is possible to display. And as well the last one. 
    
    # This is for now, it need ot be more robust 
    start_data_sample = data_sync - vid_sync
    end_frame = min(n_samples-start_data_sample-1, v_length)
    print(f"End frame: {end_frame}")

    # Preparing space for the full plot
    plot_height = 300
    plot_width = v_width
    plot_frame = np.zeros((300,plot_width,3),np.uint8)

    plot_data = (end_frame - start_data_sample)
    data_step = 1
    data_pixel = plot_width - 20
    pixel_step = int(data_pixel / plot_data)

    if pixel_step < 1:
        # we need to make data step bigger
        data_step = int(math.ceil(plot_data / data_pixel))
        pixel_step = 1

    data_pixel = int((plot_data / data_step) * pixel_step) 
    pixel_step_f = data_pixel / plot_data
    plot_x0 = int((plot_width - data_pixel) / 2)
    print(f"Plot spec. pix stp:{pixel_step}, data stp:{data_step}, plot dta:{plot_data}")

    # Plotting the full plots in the created frame
    data_point = start_data_sample
    px = plot_x0
    for _ in range(int(plot_data/data_step)):
        y0 = int(plot_height / 2)
        y1 = int(y0 - data[data_point,1] * 100)
        y2 = int(y0 - data[data_point+data_step,1] * 100)
        x1 = int(px)
        x2 = int(px + pixel_step_f * data_step)
        
        cv2.line(plot_frame, (x1,y1), (x2,y2), (255,255,0), 1)
        data_point += data_step
        px += pixel_step_f * data_step


        



    while True:
        fps = int(1 / (t_end - t_start))
        t_start = time.time()

        if (frm_step > 0 and frm == len(vid_buffer) and total_frame < end_frame) or first:
            first = False
            ret_val, frame = cap.read()
            vid_buffer.append(frame)

            if total_frame == 0:
                frame_mem_size = sys.getsizeof(vid_buffer[-1])
                print(f"Frame memory size: {frame_mem_size} Bytes")
                buffer_size = int( buffer_mem / (frame_mem_size / (1024*1024)) )
                print(f"Buffer set to {buffer_size} frames")


            total_frame += frm_step 
            buffer_len = len(vid_buffer)

            if buffer_len > buffer_size:
                del vid_buffer[0]

        buffer_len = len(vid_buffer)
        if frm >= buffer_len:
            frm = buffer_len-1


        display_frame = copy.copy(vid_buffer[frm])
        plot_frame_full = copy.copy(plot_frame)
        # plot_frame = np.zeros((300,v_width,3),np.uint8)

        # the progress bar stuff
        abs_frm = total_frame+frm-buffer_len+1

        # progress bar
        cv2.rectangle(display_frame, (11,v_height - 19), (int(11 + (v_width-22)*abs_frm/v_length), v_height-11), (125,125,125), -1) 

        # buffer bar
        cv2.rectangle(display_frame, (int(11 + (v_width-22)*(total_frame - buffer_len)/v_length),v_height - 24), (int(11 + (v_width-22)*total_frame/v_length), v_height-20), (0,0,255), -1) 

        # progress bar frame
        cv2.rectangle(display_frame, (10,v_height - 25), (v_width-10, v_height-10), (255,255,255), 1) 

        # Using cv2.putText() method
        txt_string = f"{mark[frm_step]} AbsFrame: {abs_frm}  BufferFrm: {frm} HeadPos: {total_frame} FPS: {fps}"

        # Placing play head on the plot window
        play_head_x = round(plot_x0 + abs_frm * pixel_step_f)
        cv2.line(plot_frame_full, (play_head_x,0), (play_head_x,plot_height), (0,0,255), 1)

        prev_frm = frm
        frm += frm_step
        if step:
            frm_step = 0
            step = False

        if frm < 0:
            frm = 0
            frm_step = 0


        
        image = cv2.putText(display_frame, txt_string, org, font, 
                           fontScale, color, thickness, cv2.LINE_AA)

        # Stacking images arrays 
        # display = np.vstack((display_frame, plot_frame))
        cv2.imshow('Video Frame', display_frame)
        cv2.imshow('Plot Frame', plot_frame_full)
        # cv2.imshow('Stacked', display)

        the_pressed_key = cv2.waitKey(1)
        if  the_pressed_key == 27:
            break  # esc to quit

        elif the_pressed_key == ord('j'):
            frm_step = -1

        elif the_pressed_key == ord('k'):
            frm_step = 0

        elif the_pressed_key == ord('l'):
            frm_step = 1

        elif the_pressed_key == ord('J'):
            frm_step = -1
            step = True

        elif the_pressed_key == ord('L'):
            frm_step = 1
            step = True

        t_end = time.time()

    cv2.destroyAllWindows()

def get_csv_data(csv_file, skip=8, delimiter=';'):

    with open(csv_file, 'r', newline = '') as file:

        reader = csv.reader(file, delimiter = delimiter)
        row_cnt = 0

        data_set = []
        for row in reader:
            row_cnt += 1

            data_row = []
            if row_cnt > skip:
                # processing the data row.
                if len(row) > 0:
                    for d in row:
                        temp_data = d.replace(',','.') 
                        try:
                            temp_data = float(temp_data)
                        except Exception as e:
                            temp_data = 0

                        data_row.append((temp_data))

                data_set.append(data_row)
            else:
                print("skipped: ",row)


    data_set = np.array(data_set)
    return data_set

def normalize(data_array):

    in_samples, n_cols = data_array.shape

    for col in range(n_cols):
        amplitude = max(abs(data_array[:,col].max()),abs(data_array[:,col].min()))
        print(f"Column {col}, Amplitude {amplitude}")
        if amplitude != 0:
            data_array[:,col] /= amplitude
            np.append(data_array[:,col], amplitude)
        else:
            np.append(data_array[:,col], 0)
    return data_array




def main():
    data = get_csv_data('/Users/tymancjo/LocalGit/video/sc_data_example/11435.txt')
    data = normalize(data)
    video_file = '/Users/tymancjo/LocalGit/video/sc_data_example/11435.mp4'

    play_videoFile(video_file,data,500,500)

if __name__ == '__main__':
    main()