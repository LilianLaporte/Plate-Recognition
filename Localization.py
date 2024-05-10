import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
In this file, you need to define plate_detection function.
To do:
    1. Localize the plates and crop the plates
    2. Adjust the cropped plate images
Inputs:(One)
    1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
    type: Numpy array (imread by OpenCV package)
Outputs:(One)
    1. plate_imgs: cropped and adjusted plate images
    type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
    1. You may need to define other functions, such as crop and adjust function
    2. You may need to define two ways for localizing plates(yellow or other colors)
"""

def segmentation_mask(image, plots = False,
                      color_min = np.array([10, 70, 40]),
                      color_max = np.array([27, 255, 255])):

    # Convert to HSI/HSV
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define color range in the parameters
    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    mask = cv2.inRange(img_hsv, color_min, color_max)

    if plots:
        plt.figure('input image')
        plt.imshow(image, vmin=0, vmax=255)
        plt.figure('hsv image')
        plt.imshow(img_hsv, vmin=0, vmax=255)
        plt.figure('mask')
        plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
        plt.figure('masked input image')
        plt.imshow(cv2.cvtColor(cv2.bitwise_and(img_hsv, img_hsv, mask=mask), cv2.COLOR_HSV2RGB), vmin=0, vmax=255)
        plt.show()
    return mask


def denoise(img, structuring_element):
    eroded = cv2.erode(img, structuring_element)
    return cv2.dilate(eroded, structuring_element)

def closing(img, structuring_element):
    return cv2.dilate(img, structuring_element)

def morphology_treatment(mask ,plots = False):
    dmask = denoise(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    mask = closing(dmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    if plots:
        plt.imshow(dmask, cmap='gray', vmin=0, vmax=255)
        plt.show() #denoised mask
        plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
        plt.show() #final, denoised and closed mask
    return mask

def objects_extraction(mask):
    noRegions, labelled_mask = cv2.connectedComponents(mask) #labels each conected region with an integer
    bboxes = np.zeros((noRegions - 1, 4)) #an array whose elements are bounds for regions in the mask
                                          #example element: [1, 3, 4, 5] ->
                                          #highlights a region with xmin = 1, xmax = 3, ymin = 4, ymax = 5
    for i in range(1, noRegions):
        temp = np.where(labelled_mask == i, 1, 0) #only mark one region with ones
        rows = np.any(temp, axis=1)  # 1d array with True if the row has any non zero value
        cols = np.any(temp, axis=0)  # 1d array with True if the column has any NZ values
        # select first and last indices only with [[0,-1]]
        xmin, xmax = np.where(rows)[0][[0, -1]]
        ymin, ymax = np.where(cols)[0][[0, -1]]
        bboxes[i-1]= np.array([xmin, xmax, ymin, ymax])   #x, y in matrix axes (i and j)
    return bboxes.astype(np.int64), labelled_mask

def cropping(mask, img):
    rows = np.any(mask, axis=1) #1d array with True if the row has any non zero value
    cols = np.any(mask, axis=0) #1d array with True if the column has any NZ values

    r = np.where(rows)[0] # indices of the rows with NZ values
    c = np.where(cols)[0] # indices of the cols with NZ values

    # select first and last indices only with [[0,-1]]
    xmin, xmax = r[[0, -1]] if len(r)!=0 else (0, 0)
    ymin, ymax = c[[0, -1]] if len(c)!=0 else (0, 0)

    return img[xmin:xmax+1, ymin:ymax+1]

def discardObjects(mask, bounds, regions, plots = False, eps = 0.000001):
    # it is known that plates are all at least 90 pixels wide
    # the plates can be rotated by +-40°
    # the plates are all at least 65 pixels wide
    # aspect ratio Dx/Dy \in [0.15, 1.5]
    out_mask = mask.copy()
    for i, bound in enumerate(bounds):
        dy = np.abs(bound[2]-bound[3])
        dx = np.abs(bound[0]-bound[1])
        ar = dx/(dy+eps)
        if dy < 65 or dx < 13 or ar < 0.15 or ar > 1.5:
            out_mask = np.where(regions == i+1, 0, out_mask)
            #out_mask[bound[0]:bound[1]+1, bound[2]:bound[3]+1] = 0 #faster but fucks up with overlap

    if plots:
        plt.imshow(out_mask, cmap = 'gray')
        plt.title('sorted mask')
        plt.show()

    return out_mask

def discard_by_contour(mask, thresh = 0.6, plots = False):
    debug = cv2.cvtColor(np.where(mask!=0, 255, 0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    noRegions, labelled_mask = cv2.connectedComponents(mask)
    for i in range(1, noRegions):

        region = np.where(labelled_mask==i, 255, 0).astype(np.uint8)
        cnt, h = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnt_area = cv2.contourArea(cnt[0])
        rect = cv2.minAreaRect(cnt[0])
        rect_area = rect[1][0]*rect[1][1]
        ratio = cnt_area/rect_area

        if ratio < thresh:  #true contours is too small against rotated rectangle contour ->not a rectangle
            labelled_mask = np.where(labelled_mask==i, 0, labelled_mask)

        if plots:
            debug = cv2.drawContours(debug, [np.int0(cv2.boxPoints(rect))], 0, (0,255,0),2)
            debug = cv2.drawContours(debug, cnt[0], -1, (255,0,0),2)
    if plots: plt.imshow(debug), plt.show()
    return np.where(labelled_mask != 0, mask, 0).astype(np.uint8)

def discard_by_area(mask, nbToKeep = 1, plots = False):
    result = np.where(mask!=0, 255, 0)
    noRegions, labelled_mask = cv2.connectedComponents(mask)  # labels each conected region with an integer
    if noRegions > 1:
        areas = np.array([[i, len(np.where(labelled_mask == i)[0])] for i in range(1, noRegions)])
        sorted_areas = areas[np.argsort(areas[:, 1])]

        for index in sorted_areas[:(noRegions-1-nbToKeep),0]:
            result = np.where(labelled_mask == index, 0, result)

        if plots:
            print(sorted_areas)
            plt.figure('result')
            plt.imshow(result)
            plt.figure('labels')
            plt.imshow(labelled_mask)
            plt.show()
        return result.astype(np.uint8)
    return mask

def filtering(img, kernel):
    image_filt = cv2.filter2D(img, -1, kernel)
    return image_filt

def get_mask_angle(mask, plots = False):
    #the mask should be a 1channel grayscale uint8 image
    cnt, H = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnt)>=1:
        rect = cv2.minAreaRect(cnt[0])
        boxPoints = cv2.boxPoints(rect)
        rightMost = boxPoints[np.argmax(boxPoints[:,0])]
        dists = np.array([np.linalg.norm(np.array(rightMost)-np.array(boxPoint)) for boxPoint in boxPoints])
        secondFarthest = boxPoints[np.argsort(dists)][2]
        dx = rightMost[0]-secondFarthest[0]
        dy = rightMost[1]-secondFarthest[1]
        angle = np.arctan2(dy, dx)

        if plots:
            plt.imshow(cv2.drawContours(cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR),
                                        [np.int0(cv2.boxPoints(cv2.minAreaRect(cnt[0])))], 0,(0,255,0), 2))
            plt.title(str(angle)+'rad    '+str(angle*180/np.pi)+'°')
            plt.show()
        return  angle*180/np.pi
    return int(0)

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def check_validity(mask):  #center is a tuple of x, y coordinates (axis 1, and axis 0)
    validity = True
    x, y, w, h = cv2.boundingRect(mask)
    if x<=1 or y<=1 or (x+w)>=mask.shape[1]-1 or (y+h)>=mask.shape[0]-1:
        validity = False
    return validity

def plate_detection(image, plots = False):
    #input image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get Segmentation mask
    mask = segmentation_mask(image)
    if plots:   plt.imshow(mask), plt.show()

    # Apply Morphology Process to mask
    mask = morphology_treatment(mask)
    if plots: plt.imshow(mask), plt.show()

    # Separate different objects in the mask
    bounds, labelled_mask = objects_extraction(mask)

    #Sort the objects in the mask
    mask = discardObjects(mask, bounds, labelled_mask)
    if plots: plt.imshow(mask), plt.title('size sorted'), plt.show()
    mask = discard_by_contour(mask)
    if plots: plt.imshow(mask), plt.title('contour sorted'), plt.show()
    mask = discard_by_area(mask)
    if plots: plt.imshow(mask), plt.title('area_sorted'), plt.show()

    # Get the plate's orientation with its mask
    angle = get_mask_angle(mask)

    #if the mask touches the edge, discard the analysis
    valid = check_validity(mask)

    if valid:
        # Apply the segmentation mask and rotate the plate
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        rmasked_img = rotate_image(masked_img, angle)
        rimage = rotate_image(image, angle)

        # Crop the masked image to only contain the plate
        plate = cropping(rmasked_img, rimage)
        if plots: plt.imshow(plate), plt.show()

        # Second segmentation
        mask2 = segmentation_mask(plate, color_min=np.array([10,100,100]),
                                         color_max=np.array([30,200,200]))
        masked2 = cv2.bitwise_and(plate, plate, mask = mask2)
        plate = cropping(masked2, plate)
        if plots: plt.imshow(plate), plt.show()

        if plate.shape[0] > 2 and plate.shape[1]>2:
            plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
            return mask, plate
    return None, None

def edge_detection(image, low = 70, high = 150):
    edge_image = cv2.Canny(image, low, high)
    lines, regions = objects_extraction(edge_image)
    filtered_lines = discardObjects(edge_image, lines, regions)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    FLINES = np.log(np.abs(np.fft.fftshift(np.fft.fft2(grayscale)))+0.000001)
    FLINES = (FLINES-np.amin(FLINES))*255/(np.amax(FLINES)-np.amin(FLINES)) #CONTRAST STRECHT THE FFT TO ACTUALLY SEE IT

    return edge_image, filtered_lines, FLINES
