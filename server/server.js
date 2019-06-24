const express = require('express');
const app = express();

const port = process.env.PORT || 4000;
const localhost = 'http://localhost'
app.set('view engine', 'pug');

app.use(express.static('public'));
app.use(express.urlencoded({extended: true}));

// routes
app.use(require("../routes/main.js"));

const main = async () => {
    const server = await app.listen(port);
    io = require('socket.io').listen(server);
    await require('../models/channel').io(io);
    await console.debug(`🚀  Server listening on ${localhost}:${port}`);
}

main();