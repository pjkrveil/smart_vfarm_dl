
def data_generator():	
	train_datagen = ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(
	        '../dataset/train',
	        target_size=(130, 130),
	        batch_size=5,
	        class_mode='categorical')

	test_datagen = ImageDataGenerator(rescale=1./255)

	test_generator = test_datagen.flow_from_directory(
	        '../dataset/test',
	        target_size=(130, 130),
	        batch_size=5,
	        class_mode='categorical')

	return train_generator, test_generator