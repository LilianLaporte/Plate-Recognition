import cv2
import numpy as np
import matplotlib.pyplot as plt
from Localization import objects_extraction, discard_by_area

"""
In this file, you will define your own segment_and_recognize function.
To do:
	1. Segment the plates character by character
	2. Compute the distances between character images and reference character images
	   (in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
	3. Recognize the character by comparing the distances
Inputs:(One)
	1. plate_imgs: cropped plate images by Localization.plate_detection function
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
	1. recognized_plates: recognized plate characters
	type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
	You may need to define other functions.
"""
def segment_characters(plate, c_value = 4, plots = False):
	gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

	mean_size = 2*(plate.shape[0]//10)+1 #take approximately 1/5 of the vertical size
	if mean_size <=1: mean_size=3
	thresh_plate = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
											 cv2.THRESH_BINARY_INV, mean_size, c_value)
	if plots:
		plt.figure('original'), plt.imshow(plate)
		plt.figure('threhed'), plt.imshow(thresh_plate)
		plt.show()
	return thresh_plate

def denoise(img, structuring_element):
	eroded = cv2.erode(img, structuring_element)
	return cv2.dilate(eroded, structuring_element)

def closing(img, structuring_element):
	return cv2.dilate(img, structuring_element)

def recognition_morphology_treatment(mask, plots = False):
	size = mask.shape[0]//10
	if size < 1: size = 1
	strE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
	dmask = cv2.dilate(mask, strE)
	fmask = cv2.erode(dmask, strE)

	if plots:
		plt.figure('original')
		plt.imshow(mask)
		plt.figure('dilated mask')
		plt.imshow(dmask, cmap='gray', vmin=0, vmax=255)
		plt.figure('dilated and eroded')
		plt.imshow(fmask, cmap='gray', vmin=0, vmax=255)
		plt.show()  # final, denoised and closed mask
	return fmask

def discardNonDigits(mask, bounds, regions, plots = False):
	# plates have a 1:5 aspect ratio
	out_mask = mask.copy()
	for i, bound in enumerate(bounds):
		#normalized sizes
		dyn = np.round(np.abs(bound[2]-bound[3])/mask.shape[1], 3) #axes of a matrix (x is downwards and y is leftbound)
		dxn = np.round(np.abs(bound[0]-bound[1])/mask.shape[0], 3)

		#eliminate small objects and large ones
		if dyn > 0.3 or dyn < 0.02 or dxn < 0.035:
			out_mask = np.where(regions == i+1, 0, out_mask)
	if plots:
		plt.figure('original')
		plt.imshow(mask, cmap = 'gray')
		plt.figure('sorted mask')
		plt.imshow(out_mask, cmap = 'gray')
		plt.show()
	return out_mask

def sift_descriptor(image, eps=0.000001):
	# 16 cells, 8bins histograms -> 128d descriptor
	result = np.zeros(128)
	# get the gradient of the image
	image = np.float64(image)
	sobelx = np.array([[1, 0, -1],
					   [2, 0, -2],
					   [1, 0, -1]])
	sobely = np.array([[1, 2, 1],
					   [0, 0, 0],
					   [-1, -2, -1]])

	gx = cv2.filter2D(image, -1, sobelx).astype(np.float64)
	gy = cv2.filter2D(image, -1, sobely).astype(np.float64)
	g = np.sqrt(gx ** 2 + gy ** 2).astype(np.float64)
	theta = np.angle(gx + 1j * gy)

	# Take only 16x16 window of the picture from the center
	cell = -1
	# Iterate over every pixel
	for i in range(4):  # four vertical cells
		for j in range(4):  # four horizontal cells
			cell += 1
			# each cell is 4x4 pixels
			for k in range(4):
				for l in range(4):
					xc = 4 * i + k
					yc = 4 * j + l
					gc = np.abs(g[xc, yc])
					tc = theta[xc, yc]

					if np.pi / 4 > tc >= 0:  # quadrant 1, 1
						result[8 * cell] += gc
					elif np.pi / 2 > tc >= np.pi / 4:  # 1, 2
						result[8 * cell + 1] += gc

					elif 3 * np.pi / 4 > tc >= np.pi / 2:  # quadrant 2, 1
						result[8 * cell + 2] += gc
					elif np.pi >= tc >= 3 * np.pi / 4:  # 2, 2
						result[8 * cell + 3] += gc

					elif -np.pi <= tc <= -3 * np.pi / 4:  # quadrant 3, 1
						result[8 * cell + 4] += gc
					elif -3 * np.pi / 4 < tc <= -np.pi / 2:  # 3, 2
						result[8 * cell + 5] += gc

					elif -np.pi / 2 <= tc < -np.pi / 4:  # quadrant 4, 1
						result[8 * cell + 6] += gc
					elif -np.pi / 4 < tc < 0:  # 4, 2
						result[8 * cell + 7] += gc

	return result / (np.linalg.norm(result)+eps)

def L2Norm(a, b):
	return np.linalg.norm(np.array(a)-np.array(b))

def diff_score(char, reference_char):
	xor = cv2.bitwise_xor(char, reference_char)
	return np.sum(xor)/255 #amount of wrong pixels

def xor_score(char, chars_refs, prints = False):
	scores = np.array([diff_score(char, chars_refs[k]) for k in chars_refs])
	char_list = np.array([c for c in chars_refs])
	low_scores_indices = np.argsort(scores)
	low_scores = scores[low_scores_indices]
	if prints: print(char_list[low_scores_indices], low_scores)
	ratio = low_scores[1]/low_scores[0]
	return (char_list[low_scores_indices])[:2], ratio

def NN_SIFT_Classifier(char, database, plots = False):

	des = sift_descriptor(char)
	if plots: plt.imshow(char), plt.show()

	similarities = [(L2Norm(des, value), key) for key, value in database.items()]
	similarities = np.array(similarities, dtype = [('distance', np.float64), ('label', 'S10')])

	sorted_sim = np.sort(similarities, order = 'distance')
	if plots: print(sorted_sim)

	ratio = sorted_sim[1][0]/sorted_sim[0][0]
	#return 2 best guesses
	return sorted_sim[0][1].decode('UTF-8'), sorted_sim[1][1].decode('UTF-8'), ratio

def recognize_digits(characters, database, ref_chars, delta = 1.2, plots = False):
	final_sequence = ''
	for i, char in enumerate(characters):
		char = cv2.resize(char, (16, 16))

		(xg1, xg2), xratio = xor_score(char, ref_chars)
		g1, g2, sratio = NN_SIFT_Classifier(np.array(char), database)

		if g1 == '5temp': g1 = '5'
		if g2 == '5temp': g2 = '5'
		if xg1 == '5temp': xg1 = '5'
		if xg2 == '5temp': xg2 = '5'

		if g1 == xg1: final_sequence += g1
		elif xratio > delta*sratio: final_sequence += xg1
		else: final_sequence += g1

		if plots:
			print(g1, g2, xg1, xg2)
			print(sratio, xratio)
			plt.imshow(char), plt.show()

	return final_sequence

def check_chunks_validity(sequence):
	valid = True
	chunks = sequence.split('-')
	for chunk in chunks:
		if len(np.unique(np.char.isnumeric(list(chunk)))) > 1:
			valid = False
	return valid

def segment_and_recognize(plate, database, ref_chars, prints = False):
	if plate.shape[0] > 10 and plate.shape[1] > 10:
		#adaptive thresholding on the plate
		thresh_plate = segment_characters(plate)

		#size and area discrimination of thresholded objects
		bounds, labelled_plate = objects_extraction(thresh_plate)
		sorted_by_size = discardNonDigits(thresh_plate, bounds, labelled_plate)
		sorted_plate = discard_by_area(sorted_by_size, nbToKeep=6).astype(np.uint8)

		#characters extraction
		char_bounds, ls_plate = objects_extraction(sorted_plate)
		scb = char_bounds[np.argsort(char_bounds[:, 3])] #sorted char bounds

		#lists and not np.array otherwise, numpy complains
		characters = [sorted_plate[scb[i,0]:scb[i,1]+1, scb[i,2]:scb[i,3]+1] for i in range(len(scb))]
		sequence = recognize_digits(characters, database, ref_chars)


		intervals = np.array([[i, scb[i+1, 2]-scb[i, 3]] for i in range(len(scb)-1)])
		if len(intervals) > 1:
			sorted_ints = intervals[np.argsort(intervals[:,1])]
			dashes = np.sort(np.array([sorted_ints[-1, 0]+1, sorted_ints[-2, 0]+1])) #positions of the dashes in sequence

			if len(sequence[dashes[0]:dashes[1]]) >= 2:
				sequence = sequence[:dashes[0]]+'-'+sequence[dashes[0]:dashes[1]]+'-'+sequence[dashes[1]:]

				if len(sequence) == 8 and check_chunks_validity(sequence):
					return thresh_plate, sorted_by_size, sorted_plate, sequence #recognized digits of the plate

	return None, None, None, None