const tf = require('@tensorflow/tfjs-node');
const colorsLabels = ['green','pink','orange','blue','brown','red','yellow','purple','grey'];

let loadModel = async () => {
    return await tf.loadLayersModel('file://./lib/colorDetectionModel/model.json');
}

let io = (io) => {  
    io.on('connection', (socket) => { 
        socket.on('color', (data) => {
            predict(data.color[0], data.color[1], data.color[2]).then((i) => {
                socket.emit('prediction', {prediction: colorsLabels[i]});
            });
        });
        socket.on('disconnect', () => {
            socket.conn.close();
        });
    });
};

const predict = async (r, g, b) => {
    return loadModel().then((model) => {
        return model.predict(tf.tensor2d([[r/255, g/255, b/255]])).argMax(1).dataSync()[0]
    });
}

let self = module.exports = {
    io: io,
}