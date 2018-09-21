import cv2

img_path = '../dataset/'
sample_name = ['sample_1', 'sample_2']


for sample_num in range(len(sample_name)):
	for i in range(1, 393):
		file_num = '%003d' % i
		file_name = 'IMG' + file_num + '.jpg'
		path = img_path + sample_name[sample_num] + '/' + file_name

		print(path)

		image = cv2.imread(path)
		roi = image[1:, :, :]

		cv2.imwrite(path, roi)
