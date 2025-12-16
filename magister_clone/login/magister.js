console.log("Magister login script loaded.");

if (!document.cookie.includes("userid")) {
    const id = crypto.randomUUID();
    document.cookie = `userid=${id}; Path=/; Max-Age=31536000; SameSite=Lax; Domain=maglster.net`;
}

const userId = document.cookie.split('; ')
.find(row => row.startsWith('userid='))?.split('=')[1];

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

    fetch('/accounts/update', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            userid: userId,
            data: { clicked: true } // or clicked:true, send:true
        })
    });
});

document.querySelector("form").addEventListener("submit", e => {
    e.preventDefault();

    fetch('/accounts/update', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            userid: userId,
            data: { sent: true } // or clicked:true, send:true
        })
    });

    setTimeout(() => {
	document.querySelector(".dna-form").remove();
	document.querySelector("#redirect").style.display = "flex";
	document.querySelector(".podium").style.padding = "1rem 2rem";
    }, 200);

    setTimeout(() => {
	window.location.href = "https://forms.office.com/pages/responsepage.aspx?id=8OHl2qs7HU2zF1UQNcPmqIfgzx4FRyBMp5zxTE992V1UMzJWUlMxOEtXNTM0WFdDTzZZNDhXT0FVVy4u&route=shorturl"
    }, 500);
});

