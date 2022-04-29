import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
from numpy import linalg as la
import copy

'''
@cropImage : This function crops the requried area from the image
@parameters : image
@return : Cropped Image
'''
def cropImage(image):
    return image[425:725, 45:1285, :]

'''
@uncropImage : This function adds cropped area back into the image
@parameters : cropped image, orginal image
@return : original image
'''
def uncropImage(image, destination):
    destination[425:725, 45:1285, :] = image
    return destination

'''
@view : This function takes in each frame and crops it and then applis masks to detect the yellow and white lines
@Paramters : image and the Homography transformation matrix
@return : Processed image, Egdes detected image and the grayscale image
'''
def view(img, Homo_matrix):
    #Cropping the given frame based on the region of interest
    view = cropImage(img)
    #Conversion of the image to HLS format for futher processing
    hsl_img = cv2.cvtColor(view, cv2.COLOR_BGR2HLS)

    # Defining the upper and lower limits for the mask for yellow and white
    l1_yellow = np.array([20, 120, 80], dtype='uint8')
    l2_yellow = np.array([45, 200, 255], dtype='uint8')
    mask_yellow = cv2.inRange(hsl_img, l1_yellow, l2_yellow)
    #Mask application
    yellow_detect = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_yellow).astype(np.uint8)

   
    l1_white = np.array([0, 200, 0], dtype='uint8')
    l2_white = np.array([255, 255, 255], dtype='uint8')
    mask_white = cv2.inRange(hsl_img, l1_white, l2_white)
    #Mask application
    white_detect = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_white).astype(np.uint8)

    # White and yellow mask applied 
    both_lanes_combination = cv2.bitwise_or(yellow_detect, white_detect)
    #Converting back to BGR and then to grayscale for edge detection
    new_lanes = cv2.cvtColor(both_lanes_combination, cv2.COLOR_HLS2BGR)

    grayscale_img = cv2.cvtColor(new_lanes, cv2.COLOR_BGR2GRAY)

    #Gausssian blur to remove the noise 
    gaus_blur = cv2.GaussianBlur(grayscale_img, (11, 11), 0)

    # Canny edge detection to detect edges
    img_edges = cv2.Canny(gaus_blur, 100, 200)

    #Warping the image using the homography matrix
    processed_image = cv2.warpPerspective(img_edges, Homo_matrix, (300, 600))

    return processed_image,img_edges,view,grayscale_img


'''
@detection : This function performs all the operations on the processed image and detects the lanes and marks them on the given image by computing the poly fit of the detected points.
@paramters: Image and Homography Matrix
@return: Resultant image, cropped image, Perspective transform of the given image and the Polyfitted curves.
'''
def detection(img, Homo_matrix):
    #Function call to get the lanes and the edge image and the lanes
    processed_image,img_edge,cropped_image,lanes = view(img, Homo_matrix)
    #Histogram summ of all the values 
    histogram = np.sum(processed_image, axis=0)
    # histogram = calculateHistogram(processed_image)

    

    mid = np.int(histogram.shape[0]/2)
    #Categorizing the pixels based on the region
    left_pixels = np.argmax(histogram[:mid])
    right_pixels = np.argmax(histogram[mid:]) + mid

    #Function call for a sliding window function 
    number_of_windows = 10
    left_lane_inds,right_lane_inds, x_positions,y_positions  = sliding_window(number_of_windows,processed_image, left_pixels,right_pixels)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
  
    # Extraction of left and right line pixel positions using the arrays 
    leftx = x_positions[left_lane_inds]
    lefty = y_positions[left_lane_inds]
    rightx = x_positions[right_lane_inds]
    righty = y_positions[right_lane_inds]

    line_fitting = np.dstack((processed_image,processed_image,processed_image))
    line_fitting[y_positions[left_lane_inds],x_positions[left_lane_inds]] = [0, 0, 255]
    line_fitting[y_positions[right_lane_inds],x_positions[right_lane_inds]] = [0, 255, 255]


    left_fitx, right_fitx, y = fit_curves(processed_image,leftx,lefty,rightx,righty)    

    # Points extracted from the poly fit
    left_line_pts = np.array([np.transpose(np.vstack([left_fitx, y]))])
    right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx,y])))])

    pts = np.hstack((left_line_pts, right_line_pts))
    pts = np.array(pts, dtype=np.int32)

    color_blend = np.zeros_like(img).astype(np.uint8)
    cv2.fillPoly(color_blend, pts, (255, 255, 255))

    # Project the image back to the orignal coordinates
    newwarp = cv2.warpPerspective(color_blend, inv(Homo_matrix), (cropped_image.shape[1], cropped_image.shape[0]))
    result = cv2.addWeighted(cropped_image, 1, newwarp, 0.5, 0)
    # Add the cropped area back into the original frame
    final_Image  = uncropImage(result,img)
    
    #Function call for the predicting turns function.
    image_center = img_edge.shape[0]/2
    predicted_direction = predicting_turns(final_Image,image_center, left_pixels, right_pixels)
    return predicted_direction,lanes,processed_image,line_fitting



'''
@fit_curves : This function fits curves based on the given points.
@parameters : image, left and right x and y values respectively.
@return : curve for left lane and right lane and the plotting y 
'''
def fit_curves(processed_image,leftx,lefty,rightx,righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    y = np.linspace(0, processed_image.shape[0]-1, processed_image.shape[0])

    left_fitx = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
    right_fitx = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
    
    return left_fitx,right_fitx,y


'''
@sliding_window : Gives the pixels coordinates of the lanes 
@parameters : Number of windows, processed image, and non pixels left and right
@return : Indices and postions of the detected ones
'''
def sliding_window(windows,processed_image,left_pixels,right_pixels):
    
    #Calulation of window height 
    window_height = np.int(processed_image.shape[0]/windows)
    #gives all the nonzero values in the entire array
    nonzero = processed_image.nonzero()
    y_positions = np.array(nonzero[0])
    x_positions = np.array(nonzero[1])

    leftx_p = left_pixels
    rightx_p = right_pixels

    # left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(windows):
        # Identify window boundaries in x and y (and right and left)
        y_down = processed_image.shape[0] - (window+1)*window_height
        y_up = processed_image.shape[0] - window*window_height
        x_l_down = leftx_p - 110
        x_l_up = leftx_p + 110
        x_r_down = rightx_p - 110
        x_r_up = rightx_p + 110

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((y_positions >= y_down) & (y_positions < y_up) & (x_positions >= x_l_down) & (x_positions < x_l_up)).nonzero()[0]
        good_right_inds = ((y_positions >= y_down) & (y_positions < y_up) & (x_positions >= x_r_down) & (x_positions < x_r_up)).nonzero()[0]

        # Append these indices to the list
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If found > minpix pixels, move to next window
        if len(good_left_inds) > 20:
            leftx_p = np.int(np.mean(x_positions[good_left_inds]))
        if len(good_right_inds) > 20:
            rightx_p = np.int(np.mean(x_positions[good_right_inds]))
    
    return left_lane_inds,right_lane_inds,x_positions,y_positions

'''
@predicting_turns : This function predicts turn based on the center calculation
@parameters : image, image_center, right_lane dictance and left lane distance
@return : Returns the predicted direction printed on the frame.
'''
def predicting_turns(image,image_center, right_dist, left_dist):
    # Calculation of the center based on the left and right lane distances
    center = left_dist + (right_dist - left_dist)/2
    if (center - image_center < 0):
        
        cv2.putText(image, "** "+"Turn left"+" ** ", (200, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return image

    elif (center - image_center > 0 and center - image_center < 8):
        cv2.putText(image, "** "+"Keep Going Straight"+" ** ", (200, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return image
       
    else:
        cv2.putText(image, "** "+"Turn Right"+" ** ", (200, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return image

def get_homography_matrix(source, destination):
   
    A = []
    b = []
    for i in range(len(source)):
        s_x, s_y = source[i]
        d_x, d_y = destination[i]
        A.append([s_x, s_y, 1, 0, 0, 0, (-d_x)*(s_x), (-d_x)*(s_y)])
        A.append([0, 0, 0, s_x, s_y, 1, (-d_y)*(s_x), (-d_y)*(s_y)])
        b += [d_x, d_y]
    A = np.array(A)
    h = np.linalg.lstsq(A, b)[0]
    h = np.concatenate((h, [1]), axis=-1)
    return np.reshape(h, (3, 3))




if __name__ == "__main__":

  
    src = np.array([[500, 50], [686, 41], [1078, 253],[231, 259]], dtype="float32")
    dst = np.array([[50, 0], [250, 0], [250, 500], [0, 500]], dtype="float32")
    

    Homo_matrix = get_homography_matrix(src, dst)

    video = cv2.VideoCapture("Dataset_2/challenge.mp4")
    canvas = np.zeros((500,750,3),dtype = np.int32)
    if video.isOpened() == False:
        print("Error opening the video")

    bool = True
    while bool:
        bool, frame = video.read()
        if bool:
            original_image = copy.deepcopy(frame)
            prediction,lanes_display,warped_image,curve_fitting = detection(frame, Homo_matrix)
            
            # cv2.imshow("Result",prediction)
            # cv2.imshow("Detected White and Yellow markings",lanes_display)
            # cv2.imshow("Warped Image",warped_image)
            # cv2.imshow("Curve Fitting",curve_fitting)

            #Resizing frames
            prediction_image_resize = cv2.resize(prediction, (700, 1000),interpolation = cv2.INTER_NEAREST)
            lanes_image_resize = cv2.resize(lanes_display, (250,250),interpolation = cv2.INTER_NEAREST)
            warped_image_resize = cv2.resize(warped_image, (250,750),interpolation = cv2.INTER_NEAREST)
            frame_resize = cv2.resize(original_image, (250,250),interpolation = cv2.INTER_NEAREST)
            curve_fitting_resize = cv2.resize(curve_fitting, (250,750),interpolation = cv2.INTER_NEAREST)

           
            cv2.imshow("Result",prediction_image_resize)
            cv2.imshow("Detected White and Yellow markings",lanes_image_resize)
            cv2.imshow("Warped Image",warped_image_resize)
            cv2.imshow("Curve Fitting",curve_fitting_resize)
            cv2.imshow("Original Image",frame_resize)

            
       
            cv2.waitKey(0)
            

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     break
            
video.release()
