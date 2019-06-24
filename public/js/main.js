let socket = io();

const showRGBvalues = () => {
    const RGB = ['R', 'G', 'B'];
    for (let i = 0; i < RGB.length; i++) {
        let id = document.getElementById(RGB[i]);
        id.oninput = () => {
            document.getElementById(RGB[i]+'value').innerHTML = id.value;
            drawColor();
        }
    }
}

const drawColor = () => {
    let canvas = document.getElementById('colored-circle');
    canvas.style.background = `rgb(${document.getElementById('R').value},${document.getElementById('G').value},${document.getElementById('B').value})`
    socket.emit('color', {color: [document.getElementById('R').value, document.getElementById('G').value, document.getElementById('B').value]});
}

socket.on('prediction', (data) => {
    document.getElementById('prediction').style.color = `${data.prediction}`;
    document.getElementById('prediction').innerHTML = ` ${data.prediction}`;
})

window.onload = () => showRGBvalues();