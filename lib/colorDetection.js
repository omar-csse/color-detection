const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const shuffleSeed = require('shuffle-seed');

const colorsLabels = ['green','pink','orange','blue','brown','red','yellow','purple','grey'];
let colors = [];
let labels = [];
// tensors and model
let input;
let output;
let model = tf.sequential();

const initData = async () => {
    let data = JSON.parse(fs.readFileSync(__dirname + '/mlDATA.json', 'utf8'));
    await data.sort().reverse()
    data = await shuffleSeed.shuffle(data, 230)
    await data.slice(0,0.9*data.length).forEach(color => {
        colors.push([color.r/255, color.g/255, color.b/255]);
        labels.push(colorsLabels.indexOf(color.label));
    });
}

const setupModel = () => {
    let hiddenLayer = tf.layers.dense({activation:'relu', units:64, inputDim:3});
    let outputLayer = tf.layers.dense({activation:'softmax', units:9});
    model.add(hiddenLayer);
    model.add(outputLayer);
    model.compile({optimizer: tf.train.adam(0.001, 0.9, 0.999), loss: 'categoricalCrossentropy', metrics:['accuracy']})
}

const setupTensors = () => {
    let labelTensor = tf.tensor1d(labels, 'int32');
    input = tf.tensor2d(colors);
    output = tf.oneHot(labelTensor, 9);
    labelTensor.dispose();
}

const train = async () => {
    await initData();
    await setupModel();
    await setupTensors();
    await model.fit(input, output, {shuffle: true, epochs:50, stepsPerEpoch: (colors.length/100)})
    await model.save('file://./lib/colorDetectionModel')
}

train().then(() => console.log(`training is done!, and model is saved.`))
