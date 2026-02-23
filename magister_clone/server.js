const express = require('express');
const fs = require('fs');
const app = express();

app.use(express.json());

// CORS
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "https://accounts.maglster.net");
  res.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  next();
});

// load and save to data.json
function loadData() {
  try {
    return JSON.parse(fs.readFileSync('data.json', 'utf8'));
  } catch {
    return {};
  }
}

function saveData(data) {
  fs.writeFileSync('data.json', JSON.stringify(data, null, 2));
}

app.post('/update', (req, res) => {
  const userId = req.body.userid;
  if (!userId) return res.status(400).json({ error: "missing user id" });

  const update = req.body.data;
  console.log(update);

  const data = loadData();

  if (!data[userId]) {
    data[userId] = { opened: false, clicked: false, sent: false };
  }

  Object.assign(data[userId], update);

  saveData(data);

  res.json({ status: "ok", user: userId, newState: data[userId] });
});

console.log("Server running on http://localhost:3000");

app.listen(3000);
