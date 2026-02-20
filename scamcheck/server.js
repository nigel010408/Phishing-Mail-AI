require("dotenv").config();
const express = require("express");
const session = require("express-session");
const { google } = require("googleapis");
const axios = require("axios");
const path = require("path");

const app = express();
app.use(express.json());
app.use(express.static("public"));
app.use(express.static(__dirname));

app.use(session({
    secret: process.env.SESSION_SECRET,
    resave: false,
    saveUninitialized: false
}));

const oauth2Client = new google.auth.OAuth2(
    process.env.GOOGLE_CLIENT_ID,
    process.env.GOOGLE_CLIENT_SECRET,
    process.env.GOOGLE_REDIRECT_URI
);

app.get("/login", (req, res) => {
    const url = oauth2Client.generateAuthUrl({
        access_type: "offline",
        scope: ["https://www.googleapis.com/auth/gmail.readonly"]
    });
    res.redirect(url);
});

app.get("/oauth2callback", async (req, res) => {
    const { tokens } = await oauth2Client.getToken(req.query.code);
    console.log("Tokens:", tokens);
    req.session.tokens = tokens;
    res.redirect("/");
});

app.get("/emails", async (req, res) => {
    if (!req.session.tokens) return res.status(401).send("Not logged in");

    oauth2Client.setCredentials(req.session.tokens);
    const gmail = google.gmail({ version: "v1", auth: oauth2Client });

    const list = await gmail.users.messages.list({
        userId: "me",
        maxResults: 5
    });

    const results = [];

    for (const msg of list.data.messages) {
        const full = await gmail.users.messages.get({
            userId: "me",
            id: msg.id
        });

        console.log("Email data:", full.data);

        const headers = full.data.payload.headers;
        const subject = headers.find(h => h.name === "Subject")?.value || "";
        const body = full.data.snippet || "";

        const aiResponse = await axios.post(process.env.AI_SERVER_URL, {
            subject,
            body
        });

        results.push({
            subject,
            prediction: aiResponse.data.label,
            confidence: aiResponse.data.confidence
        });
    }

    res.json(results);
});

app.listen(5000, () => console.log("Server running on port 5000"));
