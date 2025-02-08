document.addEventListener("DOMContentLoaded", function() {
    // Request access to the webcam
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            const video = document.getElementById('video-feed');
            if (video) {
                video.srcObject = stream;
            }
        })
        .catch(function(err) {
            console.error("Error accessing webcam: ", err);
        });
});