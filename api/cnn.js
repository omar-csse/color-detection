const tf = require('@tensorflow/tfjs-node');

/*  
    5.4.2 Layers API - Convolutional Neural Networks 
*/

const model = tf.sequential();
const inputLayer = tf.layers.dense({units: 16, activation:'relu', inputShape:[20,20,3]})
const convLayer1 = tf.layers.conv2d({filters: 32, kernelSize: [3,3], padding:'valid', activation:'relu'})
const poolingLayer = tf.layers.maxPooling2d({poolSize:[2,2], strides:2, padding:'valid'})
const convLayer2 = tf.layers.conv2d({filters: 64, kernelSize: [5,5], padding:'valid', activation:'relu'})
const outputLayer = tf.layers.dense({units: 2, activation: 'softmax'})

model.add(inputLayer);
model.add(convLayer1);
model.add(poolingLayer);
model.add(convLayer2);
model.add(outputLayer);