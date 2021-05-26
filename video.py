
import moviepy.editor as mvp
import pygame
 
pygame.display.set_caption('Hello World!')
#pygame.display.set_mode((0,0), 0, 32)
clip = mvp.VideoFileClip('/Volumes/TT_MAC_WIN/video/T Urbaniec/Oout/ACB_Assembly_LIne_Demo_01.mov')
clip.preview()
 
pygame.quit()