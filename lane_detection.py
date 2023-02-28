# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:59:48 2023

@author: s
"""

import cv2
import numpy as np
# Make sure the video file is in the same directory as your code
filename = 'test1.mp4'
file_size = (1280,720) 
 
# We want to save the output to a video file
output_filename = 'lane.avi'
output_frames_per_second = 20.0
def region_of_interest(image,vertices): 
    mask = np.zeros_like(image)   
    #Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #We could have used fixed numbers as the vertices of the polygon,
    #but they will not be applicable to images with different dimesnions.
    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
def draw_the_lines(img,lines):
  if lines is None:
      print("No Lines")
      return img
  imge=np.copy(img)     
  blank_image=np.zeros((imge.shape[0],imge.shape[1],3),dtype=np.uint8)
  for line in lines:
    if line is None:
        continue
    x1,y1,x2,y2 = line
    cv2.line(blank_image,(x1,y1),(x2,y2),(0,200,255),thickness=5)
    imge = cv2.addWeighted(imge,1,blank_image,1,0.0) 
  return imge
def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        return
    y1 = image.shape[0]
    y2 = int(y1*0.6)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return x1, y1, x2,y2
def average_slope_intercept(image, lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
        Parameters:
            lines: The output lines from Hough Transform.
    """
    if lines is None:
        return None
    left_fit = []
    right_fit = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope <0:
                left_fit.append((slope, intercept))
            if slope >0:
                right_fit.append((slope, intercept))
    left_fit_avg = np.average(left_fit, axis =0)
    right_fit_avg = np.average(right_fit, axis =0)
    left_line = make_coordinates(image, left_fit_avg)
    right_line = make_coordinates(image, right_fit_avg)
    return left_line, right_line

def process(image):
    #image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    height=image.shape[0]
    width=image.shape[1]
    region_of_interest_coor=[(0,height),(0,450),((3*width/4),2*height/3),(width,height)]
    image = cv2.GaussianBlur(image, (5,5), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    l_w = (0,0,200)
    u_w = (180,90,255)
    hsv = cv2.inRange(hsv, l_w, u_w)
    canny_image = cv2.Canny(hsv,100,200)
    cropped=region_of_interest(canny_image,
                              np.array([region_of_interest_coor],np.int32))
    #cropped = canny_image
    cv2.imshow("Cropped", cropped)
    blurred = cv2.GaussianBlur(cropped, (5,5),0)
    cv2.imshow("Blurred", blurred)
    lines = cv2.HoughLinesP(cropped,rho=2,theta=np.pi/120,threshold=120,lines=np.array([]),minLineLength=20,maxLineGap=100)
    lines = average_slope_intercept(image, lines)
    image_with_lines = draw_the_lines(image,lines)
    return image_with_lines

cap = cv2.VideoCapture(filename)
# Create a VideoWriter object so we can save the video output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
result = cv2.VideoWriter(output_filename,  
                           fourcc, 
                           output_frames_per_second, 
                           file_size) 
while True:
   ret, frame = cap.read() 
   if ret is False:
       cap = cv2.VideoCapture(filename)
       continue
   frame = process(frame) 
   print(cap.get(3))
   print(cap.get(4))  
   result.write(frame)
   cv2.imshow('frame', frame)    
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break 
cap.release()
result.release()
cv2.destroyAllWindows()
