document.addEventListener("DOMContentLoaded", async () => {
	try {
		const res = await fetch("http://mail.mijnkreft.site:3000/status", {
            credentials: "include"
        });
		const status = await res.json(); // ✅ need await here
        
        document.getElementById("userEmail").innerText = status.loggedIn ? status.email : "Not logged in";
    } catch (err) {
        console.error("Error checking status:", err);
        document.getElementById("userEmail").innerText = "Error fetching status";
    }
});

document.getElementById("auth").addEventListener("click", () => {
    chrome.tabs.create({ url: "http://mail.mijnkreft.site:3000/auth" });
});

document.getElementById("getEmails").addEventListener("click", async () => {
    try {
        const res = await fetch("http://mail.mijnkreft.site:3000/emails", {
            credentials: "include"
        });
        
        if (!res.ok) {
            alert("Please log in first.");
            return;
        }
        
        const emails = await res.json();
        const list = document.getElementById("emailList");
        list.innerHTML = "";
        
        emails.forEach((mail) => {
            const li = document.createElement("li");
            li.textContent = mail.from + " - " + mail.subject;
            list.appendChild(li);
        });
    } catch (err) {
        console.error(err);
        alert("Error fetching emails");
    }
});