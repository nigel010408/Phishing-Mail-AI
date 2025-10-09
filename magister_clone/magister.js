document.addEventListener("DOMContentLoaded", function() {
    const image = document.getElementById('loginImage');

    image.src = "images/login_" + (Math.floor(Math.random() * 7) + 1).toString() + ".webp";

    const date = new Date();
    const hours = date.getHours();
    let greeting;

    if (hours < 12) {
        greeting = "Goedemorgen,";
    } else if (hours < 18) {
        greeting = "Goedemiddag,";
    } else {
        greeting = "Goedenavond,";
    }

    document.getElementById('greeting').textContent = greeting;
});