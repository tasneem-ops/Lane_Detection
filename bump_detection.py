import matplotlib.pylab as plt
import cv2
import numpy as np

def region_of_interest(img,vertices): 
    mask = np.zeros_like(img) 
    #channel_count=img.shape[2]         
    match_mask_color=  255   #(255,) * channel_count
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked_image=cv2.bitwise_and(img,mask) 
    return masked_image
def draw_the_lines(img,line):
  if line is None:
      print("No Lines")
      imge=np.copy(img)     
      blank_image=np.zeros((imge.shape[0],imge.shape[1],3),dtype=np.uint8)
      cv2.putText(blank_image, "No Bump", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2,cv2.LINE_AA)
      imge = cv2.addWeighted(imge,1,blank_image,1,0.0)
      return imge
  imge=np.copy(img)     
  blank_image=np.zeros((imge.shape[0],imge.shape[1],3),dtype=np.uint8)
  x1,y1,x2,y2 = line
  cv2.line(blank_image,(x1,y1),(x2,y2),(0,200,255),thickness=5)
  cv2.putText(blank_image, "Bump", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2,cv2.LINE_AA)
  imge = cv2.addWeighted(imge,1,blank_image,1,0.0)
  return imge
def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        return
    x1 = 0
    x2 = image.shape[1]
    y1 = int(intercept)
    y2 = int(slope*x2 + intercept)
    return x1, y1, x2,y2
def average_slope_intercept(image, lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
        Parameters:
            lines: The output lines from Hough Transform.
    """
    if lines is None:
        return None
    line_fit = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            line_fit.append((slope, intercept))
    line_fit_avg = np.average(line_fit, axis =0)
    bump_line = make_coordinates(image, line_fit_avg)
    return bump_line

def process(image):
    height=image.shape[0]
    width=image.shape[1]
    region_of_interest_coor=[(0,height/3),(0,2*height/3),((width),height/3),(width,2*height/3)]
    #image = cv2.GaussianBlur(image, (3,3), 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", gray)
    #gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray,70,200)
    cropped=region_of_interest(canny_image,
                              np.array([region_of_interest_coor],np.int32))
    #cropped = canny_image
    cv2.imshow("Cropped", canny_image)
    lines = cv2.HoughLinesP(cropped,rho=2,theta=np.pi/120,threshold=255,lines=np.array([]),minLineLength=60,maxLineGap=60)
    lines = average_slope_intercept(image, lines)
    image_with_lines = draw_the_lines(image,lines) 
    return image_with_lines

image = cv2.imread("Image00307.jpg", 1)
image = process(image)
cv2.imshow("Image",image)
cv2.imwrite("output4.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()




