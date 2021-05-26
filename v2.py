import cv2
import random

def play_videoFile(filePath,mirror=False):

    cap = cv2.VideoCapture(filePath)
    cv2.namedWindow('Video Life2Coding',cv2.WINDOW_AUTOSIZE)
    while True:
        ret_val, frame = cap.read()

        if mirror:
            frame = cv2.flip(frame, 1)

        for y in range(10):
	        for x in range(10):
		        line_thickness = random.randint(1,5)
		        x1 = 10+x*192
		        x2 = x1+192
		        y1 = y*100+random.randint(1,100)
		        y2 = y*100+random.randint(1,100)

		        cv2.line(frame, (x1, y1), (x2, y2), (random.randint(50,255), random.randint(50,255), random.randint(50,255)), thickness=line_thickness)

        cv2.imshow('Video Life2Coding', frame)

        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()

def main():
    play_videoFile('/Volumes/TT_MAC_WIN/video/T Urbaniec/Oout/ACB_Assembly_LIne_Demo_01.mov',mirror=False)

if __name__ == '__main__':
    main()