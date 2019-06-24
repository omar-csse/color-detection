const tf = require('@tensorflow/tfjs-node');

/*  
    5.1 tf.tensor - function
    params:
    @ values: The values of the tensor (vector, matrix, scaler)
    @ shape - optional: The shape of the tensor (2d 1d etc.)
    @ dtype - optional: The data type of the tensor 
*/
console.log('\n\n\n5.1.1 - tensors:\n\ntf.tensor():');
const data = tf.tensor([0, 255, 189, 20], [2, 2], 'int32');
data.print();


/*  
    5.2 operations
*/

// example 1
console.log('\n\n\n5.1.2 - Operations\n\nexample 1:');
const ex1a = tf.tensor1d([1, 16]);
const ex1b = tf.tensor1d([1, 4]);
ex1a.squaredDifference(ex1b).print();
// output: [0, 144]

// example 2
console.log('\nexample 2:');
const ex2 = tf.tensor1d([1, 2, Math.E]);
ex2.log().print();  
// output: [0, 0.6931472, 0.9999999]

// example 3
console.log('\nexample 3:');
const ex3a = tf.tensor1d([1, 2]);
const ex3b = tf.tensor2d([[1, 2], [3, 4]]);
ex3a.dot(ex3b).print();
// output: [7, 10]


/*  
    5.3 Memory Management
*/

// example 1
let values = []
for (let i = 0; i < 150000; i++) {
    values[i] = Math.floor((Math.random() * 100) + 1);
} 
const shape = [500, 300];

tf.tidy(() => {
    const ex4a = tf.tensor2d(values, shape, 'int32');
    const ex4b = tf.tensor2d(values, shape, 'int32').transpose();
    let c = tf.keep(ex4a.matMul(ex4b)).print();
})


/*  
    5.4.1 Layers API - Basics
*/

// example 1 - Simple Neural Network
const model = tf.sequential();
const hiddenLayer = tf.layers.dense({units: 7, inputShape: [5]})
const outputLayer = tf.layers.dense({units: 4})
model.add(hiddenLayer);
model.add(outputLayer);
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
const input = tf.tensor2d([[0,0,0,0,0], [1,1,1,1,1]]);
const output = tf.tensor2d([[1,1,1,1], [0,0,0,0]]);
const train = async () => await model.fit(input, output, {shuffle: true, epochs:100000})
train()
    .then(() => saveModel())
    .then(() => {return loadModel()})
    .then(model => model.predict(tf.tensor2d([[0,0,0,0,0]])).print())
// output: Tensor [[0.9999964, 0.9999989, 1.0000066, 1.0000037],]



/*  
    5.5 Saving and Loading Models
*/
// example 1
const saveModel = async () => await model.save('file://./models/binaryInversion');

// example 2
const loadModel = async () => {return await tf.loadLayersModel('file://./models/binaryInversion/model.json')}
