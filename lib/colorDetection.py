import os
import json
import random
import tensorflow as tf
import numpy as np

# the collected data and labels
colorsLabels = ['green','pink','orange','blue','brown','red','yellow','purple','grey']
colors = []
labels = []
data = json.load(open(os.path.dirname(os.path.realpath(__file__))+'/mlDATA.json'))

data.sort(key=lambda x: x['r'], reverse=True)
random.seed(230)
random.shuffle(data)
trainingData = data[:int(0.9 * len(data))]
testingData = data[int(0.9 * len(data)):]
model = tf.keras.Sequential()

# initialize all the data needed for ml model
def initData():
	for color in trainingData:
		colors.append([color['r']/255, color['g']/255, color['b']/255])
		labels.append(colorsLabels.index(color['label']))

def setupModel():
	hiddenLayer = tf.keras.layers.Dense(64, activation=tf.nn.relu, input_dim=3)
	outputLayer = tf.keras.layers.Dense(9, activation=tf.nn.softmax)
	model.add(hiddenLayer)
	model.add(outputLayer)
	adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

def setupTensors():
	labelTensor = np.array(labels).reshape(-1)
	_output = np.eye(9)[labelTensor]
	_input = np.array(colors)
	return {'input': _input, 'output': _output}

def train():
	d = setupTensors()
	steps_per_epoch = int(len(colors) / 100)
	model.fit(x=d['input'], y=d['output'], epochs=50, shuffle=True, steps_per_epoch=steps_per_epoch)

def saveModel():
	model.save(os.path.dirname(os.path.realpath(__file__))+'/colorDetection.h5')

def modelAccuracy():
	# TODO
	print("Test set accuracy: ")

initData()
setupModel()
train()
saveModel()
modelAccuracy()
print('training is done!, and model is saved.')