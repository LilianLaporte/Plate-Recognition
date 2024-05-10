import argparse
import os
import cv2
from Recognize import sift_descriptor
from Localization import cropping
import CaptureFrame_Process
import numpy as np

letters_LU = np.array(['B','D','F','G',
					   'H','J','K','L',
					   'M','N','P','R',
					   'S','T','V','X','Z'])

# define the required arguments: video path(file_path), sample frequency(second), saving path for final result table
# for more information of 'argparse' module, see https://docs.python.org/3/library/argparse.html
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_path', type=str, default= 'TrainingVideo.avi')
	parser.add_argument('--output_path', type=str, default=None)
	parser.add_argument('--sample_frequency', type=int, default=2)
	args = parser.parse_args()
	return args

def database_init(directory):
	ref_chars = {}
	result = {}
	for fname in os.listdir(directory):
		if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.bmp'):
			image = cv2.imread(directory + fname)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			image = cropping(image, image)
			image = cv2.resize(image, (16,16))
			sift = sift_descriptor(image)

			if directory == './SameSizeLetters/':
				current_letter = letters_LU[int(fname.split('.')[0]) - 1]
				result[current_letter] = sift
				ref_chars[current_letter] = image
			else:
				current_number = fname.split('.')[0]
				result[current_number] = sift
				ref_chars[current_number] = image
	return result, ref_chars

# In this file, you need to pass three arguments into CaptureFrame_Process function.
if __name__ == '__main__':

	args = get_args()
	if args.output_path is None:
		output_path = os.getcwd()
	else:
		output_path = args.output_path

	path = './SameSizeNumbers/'
	database, ref_char = database_init(path)

	path = './SameSizeLetters/'
	lDatabase, ref_letters = database_init(path)

	#update the numbers database with the letters aswell
	database.update(lDatabase)      #sift database
	ref_char.update(ref_letters)	#normalized training digits on which the database is made

	file_path = args.file_path

	sample_frequency = args.sample_frequency

	CaptureFrame_Process.CaptureFrame_Process(file_path, sample_frequency, output_path, database, ref_char)



