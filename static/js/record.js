URL = window.URL || window.webkitURL;
var gumStream;

var rec;

var input;

var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext;

function startAudioRecord() {
    console.log("startRecord button clicked");


    navigator.mediaDevices.getUserMedia({
        audio: true
    }).then(function (stream) {
        console.log("Initializing Recorder.js ...");

        audioContext = new AudioContext;

        gumStream = stream;

        input = audioContext.createMediaStreamSource(stream);

        // Create the Recorder object
        rec = new Recorder(input, {
            numChannels: 1
        })
        // Start recording 
        rec.record()

    }).catch(function (err) {
        // Enable the record button if getUserMedia() fails 
console.log("catch startAudioRecord",err);
    });
}

function stopAudioRecord() {
    console.log("stopRecord button clicked");

    //tell the recorder to stop the recording 
    rec.stop(); //stop microphone access 
    gumStream.getAudioTracks()[0].stop();
    //create the wav blob and pass it on to createDownloadLink 
    rec.exportWAV(createDownloadLink);
}

function createDownloadLink(blob) {
    var url = URL.createObjectURL(blob);
    var li = document.createElement('li');
    var audioPlayback = document.createElement('audio');
    var link = document.createElement('a');
    var b = document.createElement('b');

    audioPlayback.controls = true;
    audioPlayback.src = url;
    // #######################################
    link.href = url;
    filename = new Date().toISOString();
    link.download = filename + ".wav";
    link.innerHTML = "Download";


    b.innerHTML = "Processing";
    li.appendChild(b);


    let audiodata;
    var xhr = new XMLHttpRequest();
    xhr.onload = function (e) {
        if (this.readyState === 4 && this.status == 200) {
            console.log("Server accessed successfully");
            audiodata = JSON.parse(this.responseText)
            console.log(audiodata);

//            document.location = "/audio?audioemotion=" + audiodata.audioemotion + "&audioPS=" + audiodata.audioPS +
//                "&token=" + audiodata.access_token;

        }
        else{
            alert("something went wrong in audio processing");
        }
    };

    // #######################################
    // upload.upload = filename+".wav";
    var fd = new FormData();
    fd.append("audio_data", blob, filename);
    xhr.open("POST", "/audioemotion", true);
    xhr.send(fd);


}