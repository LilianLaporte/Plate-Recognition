import cv2
import pandas as pd
import matplotlib.patches as patches
import Localization
import Recognize

#extras
import numpy as np
import matplotlib.pyplot as plt


"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
	1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
	2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segment_and_recognize' functions)
	3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
	1. file_path: video path
	2. sample_frequency: second
	3. save_path: final .csv file path
Output: None
"""

def xor_performance(mask, gt, plots = False):
    xor = cv2.bitwise_xor(mask, gt)

    if plots:
        plt.figure('GT')
        plt.imshow(gt, cmap = 'gray')

        plt.figure('mask')
        plt.imshow(mask, cmap = 'gray')

        plt.figure('xor')
        plt.imshow(xor, cmap = 'gray')
        plt.show()

    return round(np.sum(xor)/np.sum(gt), 2)

def iou_performance(mask, gt, plots = False):
    # Get gt coordinates
    index_gt = np.nonzero(gt)
    xmin_gt, xmax_gt = min(index_gt[0]), max(index_gt[0])
    ymin_gt, ymax_gt = min(index_gt[1]), max(index_gt[1])
    # Get mask coordinates
    index_mask = np.nonzero(mask)
    if len(index_mask[0]) != 0:
        xmin_mask, xmax_mask = min(index_mask[0]), max(index_mask[0])
        ymin_mask, ymax_mask = min(index_mask[1]), max(index_mask[1])
    else:
        xmin_mask, xmax_mask, ymin_mask, ymax_mask = 0,0,0,0

    # Get intersection coordinates
    xmin_inter, ymin_inter = max(ymin_gt, ymin_mask), max(xmin_gt, xmin_mask)
    xmax_inter, ymax_inter = min(ymax_gt, ymax_mask), min(xmax_gt, xmax_mask)

    # if no intersect, then we want it to be 0
    delta_x = (xmax_inter - xmin_inter) if (xmax_inter - xmin_inter) > 0 else 0
    delta_y = (ymax_inter - ymin_inter) if (ymax_inter - ymin_inter) > 0 else 0

    intersection = delta_x * delta_y
    boxgt_area = abs((xmax_gt - xmin_gt) * (ymax_gt - ymin_gt))
    boxmask_area = abs((xmax_mask - xmin_mask) * (ymax_mask - ymin_mask))

    if plots:
        figure, ax = plt.subplots(1)
        height_gt = xmax_gt-xmin_gt
        width_gt = ymax_gt-ymin_gt
        height_mask = xmax_mask-xmin_mask
        width_mask = ymax_mask-ymin_mask
        rect_gt = patches.Rectangle((ymin_gt, xmin_gt), width_gt, height_gt,
                                    edgecolor='b', facecolor="none", label = "Ground truth")
        rect_mask = patches.Rectangle((ymin_mask, xmin_mask), width_mask, height_mask,
                                      edgecolor='r', facecolor="none", label = "Mask")
        ax.imshow(mask, cmap='gray')
        ax.add_patch(rect_gt)
        ax.add_patch(rect_mask)
        plt.legend()
        plt.show()

    return round(intersection / (boxgt_area + boxmask_area - intersection + 1e-6), 2)

def set_compare(x, y): #x, y SAME SIZE sets of similar elements
    assert x.shape == y.shape
    ux = np.unique(x, return_counts = True)
    uy = np.unique(y, return_counts = True)
    diffs = 0
    for i in range(len(ux[0])):
        charX = ux[0][i]
        count_charX = ux[1][i]
        if charX in uy[0]: #charX is also in uy but in different amount
            count_charY = uy[1][np.where(uy[0]== charX)[0][0]]
            diffs += np.abs(count_charX-count_charY)
        else:
            diffs += count_charX #charX is not in uy
    return diffs

def voting(batch):
    final_sequence = ''
    for i in range(8): #8 is the length of the plate
        char_i = np.array([list(plate)[i] for plate in batch])
        uchar_i = np.unique(char_i, return_counts = True)
        final_sequence += uchar_i[0][np.argmax(uchar_i[1])] #append most occurring
    return final_sequence

def CaptureFrame_Process(file_path, sample_frequency, save_path, database, ref_chars):
    cap = cv2.VideoCapture(file_path)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    print('framerate', framerate)
    frames = []
    ret = True
    while ret:
        # Read the video frame by frame
        ret, frame = cap.read()
        frame = np.array(frame)
        if ret:
            frames.append(frame)
    # When everything done, release the capture
    cap.release()

    # Final array containing every frame
    frames = np.array(frames)

    plates = np.array([])           #container for each valid frame's recognized plate
    frame_no = np.array([])         #container for the frame of first detection
    temp_plates = np.array([])      #container to contain every guess of a same plate's appearance
    final_plates = np.array([])     #container with each single plate only once

    starting_frame = 0
    transition = False
    confirm = False

    #iterate over every frame
    for k, frame in enumerate(frames[starting_frame:]):
        if k%sample_frequency == 0: #only launch recognition at the correct sample rate

            #extracted plate
            plate = Localization.plate_detection(frame)[1]
            if plate is not None:  #only the valid plates come through

                #extracted plate number
                plate_number = Recognize.segment_and_recognize(plate, database, ref_chars)[3]
                if plate_number is not None:
                    print(plate_number)

                    #record first plate frame of appearance
                    if len(frame_no) == 0:
                        frame_no=np.append(frame_no, k)
                        last_at_transition = plate_number

                    #store the recognized plate
                    plates = np.append(plates, plate_number)
                    temp_plates = np.append(temp_plates, plate_number)


                    if len(plates) < 2: last_plate =  np.array(list(plates[-1])) #avoids out of bounds
                    else: last_plate = np.array(list(plates[-2]))

                    if transition:
                        confirm = len(np.where(np.array(list(plates[-1])) != last_at_transition)[0]) >= 4 and\
                                  (k-frame_no[-1]) >= 25 and\
                                  set_compare(np.array(list(plates[-1])), np.array(last_at_transition)) >= 2

                    #confirmed transition case
                    if transition and confirm:
                        frame_no = np.append(frame_no, frame_at_transition)
                        final_plates = np.append(final_plates, voting(temp_plates[:-2]))
                        temp_plates = temp_plates[-2:]
                        transition = False
                        confirm = False

                    #fluke
                    elif transition and not confirm: transition = False

                    #check if new possible transition
                    if len(np.where(np.array(list(plates[-1])) != last_plate)[0]) >= 4 and (k - frame_no[-1]) >= 25 and\
                                 set_compare(np.array(list(plates[-1])), np.array(last_plate)) >= 2:
                        transition = True
                        frame_at_transition = k
                        last_at_transition = last_plate

    #compute the last plate, after the last change
    final_plates = np.append(final_plates, voting(temp_plates))
    #eventually add a security margin to frame_no
    frame_no += 5

    #compute timestamps consistently with the margin
    time_stamps = np.round(frame_no / framerate, 3)

    print(final_plates)
    print(frame_no)

    df = pd.DataFrame({'License plate': final_plates, 'Frame no.': frame_no, 'Timestamp(seconds)':time_stamps})
    # noinspection PyTypeChecker

    df.to_csv(save_path, index = False)
