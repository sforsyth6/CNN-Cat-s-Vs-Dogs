import tensorflow as tf
import numpy as np
import zipfile
import os
import random
from scipy import ndimage,misc
from matplotlib import pyplot as plt
from six.moves import cPickle as pickle
from sklearn.model_selection import train_test_split

filename = 'train.zip'
folder = 'train/'
direc = 'cat-dog'
direc2 = 'doggos'
current_dir = os.path.dirname(os.path.realpath(__file__))

def unzip():
	with zipfile.ZipFile(filename, 'r') as zip:
		zip.extractall()

#this is used to take in each image from the train data files, convert them into an array of pixels,
#collects the images into groups of 2500, and then pickles them to make data retrieval more efficient
def image_converter(folder):	
	if 'train' not in os.listdir('.'):
		unzip()
	if not os.path.exists(direc):
    		os.makedirs(direc)
	
	file_names = os.listdir(folder)

	pickle_size = len(file_names)/10
	data_dog, data_cat = [np.empty(shape = (pickle_size,image_size,image_size)) for i in range(2)]
	c,d = [0 for i in range(2)]
	iteration_cat, iteration_dog = [1 for i in range(2)]
	for i ,name in enumerate(file_names):
		
		#turn the photos into an array and then set mean = 0, variance ~ 0.5, and resize to 150x150
		image = ndimage.imread(folder + name, flatten = True)
		norm_image = (image.astype(float) - 255/2)/255
		resize = misc.imresize(norm_image, size = (image_size,image_size))
		
		#seperate into groups for labeling purposes
		if 'cat' in name:
			data_cat[c, :, :] = resize
			c += 1
		elif 'dog' in name:
			data_dog[d, :, :] = resize
			d += 1

		#pickle the cat
		if c % pickle_size == 0 and c != 0:
			print (pickle_me(data_cat, 'cat', iteration_cat))
			data_cat = np.empty(shape = (pickle_size,image_size,image_size))
			iteration_cat += 1
			c = 0

		#pickle the dog
		elif d % pickle_size == 0 and d != 0:
			print (pickle_me(data_dog, 'dog', iteration_dog))
			data_dog = np.empty(shape = (pickle_size,image_size,image_size))
			iteration_dog += 1
			d = 0

	del data_dog
	del data_cat
	os.chdir(current_dir)
	return 'All the images have been pickled!'	

def pickle_me(data, category, iteration):
	#pickle the animals
	try:
		with open(direc + '/' + '%s.pickle' %(category + str(iteration)), 'wb') as f:
			pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print('Unable to save data to dog.pickle:', e)
		raise

	return "Your %s has been pickled!" % (category + str(iteration))

def vectors(data_size, image_size):
	labels = np.empty(shape = (data_size,1), dtype = np.int32)
	data = np.empty(shape = (data_size, image_size, image_size), dtype = np.float32)
	return data, labels
	
def randomize(data,labels):
	perm = np.random.permutation(labels.shape[0])
	random_data = data[perm, :, :]
	random_labels = labels[perm]
	return random_data, random_labels

#alternate the list with cat and dog pickles
def ordered_list(pickles):
	pickled = []
	while len(pickled) < len(pickles):
                if len(pickled) == 0:
                        pickled.append(pickles[0])
                else:
                        rand = random.randint(1,len(pickles)-1)
                        if 'dog' in pickled[len(pickled) - 1]:
                                if 'cat' in pickles[rand] and pickles[rand] not in pickled:
                                        pickled.append(pickles[rand])
                        elif 'cat' in pickled[len(pickled) - 1]:
                                if 'dog' in pickles[rand] and pickles[rand] not in pickled:
                                        pickled.append(pickles[rand])
	return pickled

#unpickle the cat and dog files, combine them into groups of 5000, shuffle the data,
#and then repickle for efficient access 
def unpickle(total_size, image_size):
	if not os.path.exists(direc2):
                os.makedirs(direc2)

	pickles = [word for word in os.listdir(direc) if '.pickle' in word]

	size = (total_size/ 5)
	#shuffle the list of pickled list names
	np.random.shuffle(pickles)
	
	#make the shuffled list sequential: [cat,dog,cat...]
	pickles = ordered_list(pickles)
	
	data,labels = vectors(size, image_size)
	iteration = 1
	for i,file_name in enumerate(pickles):
		with open(direc + '/' + file_name, 'rb') as f:
			try:	
				pickled_data = pickle.load(f)
				np.random.shuffle(pickled_data)
		
				shape = pickled_data.shape[0]
				count = (i % 2)*shape
				data[count:(count+shape), :, :] = pickled_data

				if 'dog' in file_name:
					#create labels for the dogs: 1 for dogs
					temp_labels = np.empty(shape=(shape,1), dtype = np.int32)
					temp_labels.fill(1)
				elif 'cat' in file_name:
					#create labels for the cats: 0 for cats
					temp_labels = np.zeros(shape=(shape,1), dtype = np.int32)
					
				labels[count: (count+shape), 0] = temp_labels[:,0]
					
			except Exception as e:
     				print('Unable to process data from ' + file_name + ':', e)
				raise      		
		
		#after getting a cat pickle, 2500, and a dog pickle, 2500, into one data file, and generating labels,
		#randomize the data, and then store it in 5 pickled files of length 5000 each
		if (i+1) % 2 == 0:
			all_data, all_labels = randomize(data,labels)
			with open(direc2 + '/' + 'doggo%i.pickle' % iteration, 'wb') as f:
				save = {'all_data': all_data, 'all_labels': all_labels}
				pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)	
			print ('doggo%i has been pickled!' % iteration)
			iteration += 1	
			data,labels = vectors(size, image_size)
	return None


#This unpickles the randomized cat + dog data each time it's called. It randomly chooses one of the 5 pickles
#and then extracts (train_size) amount of data randomly, and then repeats for (valid_size) and (test_size)
def unpickle_trained(train_size, valid_size, test_size):
	file_names = os.listdir(direc2)
	rand = random.randint(0,(len(file_names)-1))
	doggo = file_names[rand]
	with open(direc2 + '/' + doggo, 'rb') as f:
		all_stuff = pickle.load(f)

		size = all_stuff['all_data'].shape[0]
		train_offset = random.randint(0,(size - train_size))
		valid_offset = random.randint(0,(size - valid_size))
		test_offset = random.randint(0,(size - test_size))

		train_data = all_stuff['all_data'][train_offset:(train_offset+train_size), :, :]
		train_labels = all_stuff['all_labels'][train_offset: (train_offset+train_size), 0]
		index, iteration = [0 for i in range(2)]


		valid_data = all_stuff['all_data'][valid_offset:(valid_offset+valid_size), :, :]
		valid_labels = all_stuff['all_labels'][valid_offset: (valid_offset+valid_size), 0]


		test_data = all_stuff['all_data'][test_offset:(test_offset+test_size), :, :]
		test_labels = all_stuff['all_labels'][test_offset: (test_offset+test_size), 0]

	return train_data,train_labels, valid_data, valid_labels, test_data, test_labels

def reformat(dataset, labels):
        labels = (np.arange(2) == labels[:,None]).astype(np.float32)
        dataset = dataset.reshape((-1, image_size, image_size, num_in_channels)).astype(np.float32)
        return dataset, labels

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

	
#hyper-params
image_size = 150
total_size = 25000
train_size = 10
valid_size = 5
test_size = 5
num_in_channels = 1

#branch 1 patch sizes
patch_size_1 = 5
patch_size_2 = 5
patch_size_3 = 3
patch_size_4 = 3

#branch 2 patch sizes
patch_size_21 = 1
patch_size_22 = 3
patch_size_23 = 5
patch_size_24 = 3

#branch 1
num_out_channels_1 = 10
num_out_channels_2 = 30
num_out_channels_3 = 90
num_out_channels_4 = 270

#branch 2
num_out_channels_21 = 10
num_out_channels_22 = 30
num_out_channels_23 = 90
num_out_channels_24 = 270

#more hyper-params
dropout = 0.75
alpha = 0.00005
beta = 0.001
dSteps = 1000000
dRate = 0.96


#checks to see if the cat-dog files have been pickled. If not, start pickling
if 'cat-dog' not in os.listdir('.'):
	print (image_converter(folder))

#checks to see if the doggos files (the random cat and dog data) exist. If not, generate those
if 'doggos' not in os.listdir('.'):
	unpickle(total_size,image_size)

graph = tf.Graph()


training_data, training_labels, valid_data, valid_labels, test_data, test_labels = unpickle_trained(train_size, valid_size, test_size)
valid_data, valid_labels = reformat(valid_data, valid_labels)
test_data, test_labels = reformat(test_data, test_labels)

with graph.as_default():

	#declare inputs
	train_data = tf.placeholder(tf.float32, shape = (train_size,image_size,image_size, num_in_channels))
	train_label = tf.placeholder(tf.float32, shape = (train_size, 2))

	valid_data = tf.constant(valid_data)
	test_data = tf.constant(test_data)
	
	#declare variables

	#create a variable  learning rate
	global_step = tf.Variable(0)
	alpha = tf.train.exponential_decay(alpha,global_step,dSteps, dRate)
	

	#branch 1 weights and biases
	weights_1 = tf.Variable(
			tf.random_normal([patch_size_1, patch_size_1, num_in_channels, num_out_channels_1], mean = 0,  stddev=0.1))
	biases_1 = tf.Variable(tf.zeros([num_out_channels_1]))

	weights_2 = tf.Variable(
			tf.random_normal([patch_size_2,patch_size_2, num_out_channels_1, num_out_channels_2],mean = 0, stddev = 0.1))
	biases_2 = tf.Variable(tf.zeros([num_out_channels_2]))

	weights_3 = tf.Variable(
			tf.random_normal([patch_size_3,patch_size_3, num_out_channels_2, num_out_channels_3], stddev = 0.1))
	biases_3 = tf.Variable(tf.zeros([num_out_channels_3]))
	
	weights_4 = tf.Variable(
			tf.truncated_normal([patch_size_4,patch_size_4, num_out_channels_3, num_out_channels_4], stddev = 0.1))
	biases_4 = tf.Variable(tf.zeros([num_out_channels_4]))

	#fully connected weights and biases
	
	#branch 1

	weights_5 = tf.Variable(
			tf.truncated_normal([25*25*num_out_channels_4,50], stddev = 0.1))
	biases_5 = tf.Variable(tf.zeros([50]))
	
	weights_6 = tf.Variable(
			tf.truncated_normal([50,100], stddev = 0.1))
	biases_6 = tf.Variable(tf.zeros([100]))

	#branch 2
	weights_7 = tf.Variable(
			tf.truncated_normal([25*25*num_out_channels_24,50], stddev = 0.1))
	biases_7 = tf.Variable(tf.zeros([50]))
	
	weights_8 = tf.Variable(
			tf.truncated_normal([50,100], stddev = 0.1))
	biases_8 = tf.Variable(tf.zeros([100]))

	#fully connected layer for both branches
	weights_9 = tf.Variable(
			tf.truncated_normal([200,300], stddev = 0.1))
	biases_9 = tf.Variable(tf.zeros([300]))
	
	weights_10 = tf.Variable(
			tf.truncated_normal([300,2], stddev = 0.1))
	biases_10 = tf.Variable(tf.zeros([2]))


	#branch 2 weights and biases

	weights_21 =tf.Variable(
			tf.random_normal([patch_size_21, patch_size_21, num_in_channels, num_out_channels_21], mean = 0, stddev = 0.1))
	biases_21 = tf.Variable(tf.zeros([num_out_channels_21]))

	weights_22 =tf.Variable(
			tf.random_normal([patch_size_22, patch_size_22, num_out_channels_21, num_out_channels_22], mean = 0, stddev = 0.1))
	biases_22 = tf.Variable(tf.zeros([num_out_channels_22]))
	
	weights_23 =tf.Variable(
			tf.random_normal([patch_size_23, patch_size_23, num_out_channels_22, num_out_channels_23], mean = 0, stddev = 0.1))
	biases_23 = tf.Variable(tf.zeros([num_out_channels_23]))

	weights_24 =tf.Variable(
			tf.random_normal([patch_size_24, patch_size_24, num_out_channels_23, num_out_channels_24], mean = 0, stddev = 0.1))
	biases_24 = tf.Variable(tf.zeros([num_out_channels_24]))

	def model(data, train = False):
		#branch 1: (data -> 5x5 conv -> 5x5 conv -> 3x3 conv (s=3)-> 3x3 conv -> 3x3 avg_pool (s=2) -> reshape -> localConec -> localConec -> combine)
		layer1 = tf.nn.conv2d(data, weights_1, [1,1,1,1], padding = 'SAME')
		hidden = tf.nn.relu(layer1 + biases_1)
		layer2 = tf.nn.conv2d(hidden, weights_2, [1,1,1,1], padding = 'SAME')
		hidden = tf.nn.relu(layer2 + biases_2)
		layer3 = tf.nn.conv2d(hidden, weights_3, [1,2,2,1], padding = 'SAME')	
		hidden = tf.nn.relu(layer3 + biases_3)
		layer4 = tf.nn.conv2d(hidden, weights_4, [1,1,1,1], padding = 'SAME')
		hidden = tf.nn.relu(layer4)


		avg_pool = tf.nn.avg_pool(hidden, [1,3,3,1],[1,3,3,1], padding = 'SAME')
		shape = avg_pool.get_shape().as_list()
		pool = tf.reshape(avg_pool, [shape[0], shape[1]*shape[2]*shape[3]])

		local_1 = tf.nn.relu(tf.matmul(pool,weights_5) + biases_5)	
		local_2 = tf.nn.relu(tf.matmul(local_1,weights_6) + biases_6)
			
		#branch 2: (data -> 1x1 conv -> 3x3 conv -> 3x3 max_pool (s=3) -> 5x5 conv (s=2) -> 3x3 conv -> localConec -> localConec) 
		layer21 = tf.nn.conv2d(data, weights_21, [1,1,1,1], padding = 'SAME')
		hidden = tf.nn.relu(layer21 + biases_21)
		layer22 = tf.nn.conv2d(hidden, weights_22, [1,1,1,1], padding = 'SAME')	
		hidden = tf.nn.relu(layer22 + biases_22)
		pool = tf.nn.max_pool(hidden, [1,3,3,1], [1,3,3,1], padding = 'SAME')
		layer23 = tf.nn.conv2d(pool, weights_23, [1,2,2,1], padding = 'SAME')
		hidden = tf.nn.relu(layer23 + biases_23)
		layer24 = tf.nn.conv2d(hidden, weights_24, [1,1,1,1], padding= 'SAME')	
		hidden = tf.nn.relu(layer24 + biases_24)

		shape = hidden.get_shape().as_list()
		layer24 = tf.reshape(hidden, [shape[0],shape[1]*shape[2]*shape[3]])
	
		local_21 = tf.nn.relu(tf.matmul(layer24,weights_7) + biases_7)	
		local_22 = tf.nn.relu(tf.matmul(local_21,weights_8) + biases_8)
	
		#combine branches: [train_size, x], [train_size,y ] -> [train_size, x+y]
		combine = tf.concat([local_2, local_22], 1)

		fully_1 = tf.nn.relu(tf.matmul(combine,weights_9) + biases_9)

		logits = tf.matmul(fully_1, weights_10) + biases_10

		if train == True:
			tf.nn.dropout(local_1, dropout)
			tf.nn.dropout(local_2, dropout)	
			tf.nn.dropout(local_21, dropout)
			tf.nn.dropout(local_22, dropout)
			tf.nn.dropout(fully_1, dropout)

		return logits

	
	#get training logits
	logits = model(train_data, True)

	#generate training, valid, testing predictions using softmax
	train_prediction = tf.nn.softmax(logits)

	vlog = model(valid_data)
	tlog = model(test_data)
	valid_prediction = tf.nn.softmax(vlog)
	test_prediction = tf.nn.softmax(tlog)

	#Generate a loss function and then add l2 loss regularization
	loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = train_label))

	loss = loss + beta*(tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + 
				tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_4) +
				tf.nn.l2_loss(weights_5) + tf.nn.l2_loss(weights_6) +
				tf.nn.l2_loss(weights_7) + tf.nn.l2_loss(weights_8) +
				tf.nn.l2_loss(weights_9) + tf.nn.l2_loss(weights_10) + 
				tf.nn.l2_loss(weights_21) + tf.nn.l2_loss(weights_22) + 
				tf.nn.l2_loss(weights_23) + tf.nn.l2_loss(weights_24))

	#create an optimizer which minimizes loss and global_step
	optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss, global_step = global_step)

num_steps = 101

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	average = 0
	for i in range(num_steps):
		#generate data from the pickle files
		training_data, training_labels, valid_data, valid_labels, test_data, test_labels = unpickle_trained(train_size, valid_size, test_size)

		#reformat that data to make it useful for convolution
		batch_data, batch_labels = reformat(training_data, training_labels)
		valid_data, valid_labels = reformat(valid_data, valid_labels)
		test_data, test_labels = reformat(test_data, test_labels)
		
		#create feed_dict and then run the session
		feed_dict = {train_data: batch_data, train_label: batch_labels}
		l, _, predic = session.run([loss, optimizer, train_prediction],feed_dict=feed_dict)
		
		average += l
		if i % 10 == 0 and i != 0:	
			val_predic = valid_prediction.eval()
			print (i, (round(average / (i+1),2)), accuracy(predic, batch_labels), accuracy(val_predic, valid_labels))
			average = 0
	predic = test_prediction.eval() 	
	print ('Test accuracy: %.2f%%' % accuracy(predic, test_labels))		

			
