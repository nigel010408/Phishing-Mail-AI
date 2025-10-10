const express = require("express");
const { google } = require("googleapis");
const cors = require("cors");
const session = require("express-session");

const app = express();

// ✅ Correct CORS configuration
const allowedOrigins = [
  "chrome-extension://paknlomkdapiogalkjflnlkgjfoogjpf",
  "https://mijnkreft.site",
  "http://mail.mijnkreft.site:3000"
];

app.use((req, res, next) => {
  const origin = req.headers.origin;
  if (allowedOrigins.includes(origin)) {
    res.header("Access-Control-Allow-Origin", origin);
  }
  res.header("Access-Control-Allow-Credentials", "true");
  res.header("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
  
  if (req.method === "OPTIONS") {
    return res.sendStatus(200);
  }

  next();
});


app.use(session({
  secret: "supersecretkey", // change in production
  resave: false,
  saveUninitialized: false,
  cookie: { 
    secure: false, // set true if HTTPS
    sameSite: "lax"
  },
}));

// Google OAuth credentials
const CLIENT_ID = "998616122556-otre53vs47ea9drcab7opcs1i9c6d79u.apps.googleusercontent.com";
const CLIENT_SECRET = "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";
const REDIRECT_URI = "http://mail.mijnkreft.site:3000/oauth2callback";

function getOAuthClient() {
  return new google.auth.OAuth2(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI);
}

// Start login flow
app.get("/auth", (req, res) => {
  const oAuth2Client = getOAuthClient();
  const url = oAuth2Client.generateAuthUrl({
    access_type: "offline",
    scope: [
      "https://www.googleapis.com/auth/gmail.readonly",
      "email",
      "profile",
    ],
  });
  res.redirect(url);
});

// OAuth2 callback
app.get("/oauth2callback", async (req, res) => {
  try {
    const oAuth2Client = getOAuthClient();
    const { code } = req.query;
    const { tokens } = await oAuth2Client.getToken(code);

    oAuth2Client.setCredentials(tokens);

    // Fetch user info
    const oauth2 = google.oauth2({ auth: oAuth2Client, version: "v2" });
    const userInfo = await oauth2.userinfo.get();

    // Save in session
    req.session.tokens = tokens;
    req.session.user = {
      email: userInfo.data.email,
      name: userInfo.data.name,
      picture: userInfo.data.picture,
    };

    res.send("✅ Authentication complete. You can close this tab and return to your extension.");
  } catch (err) {
    console.error("OAuth error:", err);
    res.status(500).send("Authentication failed.");
  }
});

// Fetch recent emails
app.get("/emails", async (req, res) => {
  if (!req.session.tokens) return res.status(401).send("Not authenticated");

  const emails = await getEmails();

  res.json(emails);
});

// Status check
app.get("/status", (req, res) => {
  if (req.session?.user) {
    res.json({
      loggedIn: true,
      email: req.session.user.email,
      picture: req.session.user.picture,
    });
  } else {
    res.json({ loggedIn: false });
  }
});

// Logout
app.post("/logout", (req, res) => {
  req.session.destroy(() => {
    res.json({ success: true });
  });
});


async function getEmails() {
  const oAuth2Client = getOAuthClient();
  oAuth2Client.setCredentials(req.session.tokens);

  const gmail = google.gmail({ version: "v1", auth: oAuth2Client });
  const result = await gmail.users.messages.list({
    userId: "me",
    maxResults: 20,
  });

  console.log(result.data.messages);

  const messages = [];
  for (const msg of result.data.messages || []) {
    const fullMsg = await gmail.users.messages.get({
      userId: "me",
      id: msg.id,
    });

    const subject = fullMsg.data.payload.headers.find((h) => h.name === "Subject")?.value;
    const from = fullMsg.data.payload.headers.find((h) => h.name === "From")?.value;
    
    console.log(subject, from);

    messages.push({
      id: msg.id,
      from: from || "Unknown",
      subject: subject || "(No subject)",
      snippet: fullMsg.data.snippet,
    });
  }

  return messages;
}


app.listen(3000, "0.0.0.0", () => console.log("✅ Server running on port 3000"));
