# import module.py
from module import *

# Hyper Parameter
batch_size = 64
data_height = 170
data_width = 100
channel_n = 1

num_classes = 3

# # Specify class label name
# class_names = ['Healthy', 'Thres', 'Unhealthy']

# Call dataset
data_list = glob('dataset\\training\\*\\*.jpg')

# To call other images and labels, extract path from a image
path = data_list[0]
image = np.array(Image.open(path))

# onehot-encoding through label name
class_name = get_label_from_path(path)


label_name_list = []

for path in data_list:
	label_name_list.append(get_label_from_path(path))

unique_label_names = np.unique(label_name_list)

# For checking onehot_encoding
onehot_encode_label(path)

# Make a batch set
batch_image = np.zeros((batch_size, data_height, data_width, channel_n))
batch_label = np.zeros((batch_size, num_classes))

# Make a simple batch data
for n, path in enumerate(data_list[:batch_size]):
	image = read_image(path)
	onehot_label = onhot_encode_label(path)
	batch_image[n, :, :, :] = image
	batch_label[n, :] = onehot_label

print("Batch Image's shape and Batch Label's shape : ")
print(batch_image.shape, batch_label.shape)

test_n = 0
plt.title(batch_label[test_n])
plt.imshow(batch_image[test_n, :, :, 0])
plt.show()


# Generate Batch Data

batch_per_epoch = batch_size // len(data_list)

# Change label into list not for putting array type
label_list = [onehot_encode_label(path).tolist() for path in data_list]

dataset = tf.data.Dataset.from_tensor_slices((data_list, label_list))
dataset = dataset.map(
	lambda data_list, label_list:
	tuple(tf.py_func(_read_py_function, [data_list, label_list], [tf.int32, tf.uint8])))

dataset = dataset.map(_resize_function)
dataset = dataset.repeat()
dataset = dataset.shuffle(buffer_size=(int(len(data_list) * 0.4) + 3 * batch_size))
dataset = dataset.batch(batch_size)

iterator = dataset.make_initializable_iterator()
image_stacked, label_stacked = iterator.get_next()

next_element = iterator.get_next()

with tf.Session() as sess:
	sess.run(iterator.initializer)
	image, label = sess.run([image_stacked, label_stacked])