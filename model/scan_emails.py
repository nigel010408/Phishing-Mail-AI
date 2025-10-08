# scan_emails.py
import joblib
from email import policy
from email.parser import BytesParser
import imaplib

# Laad getraind model
model = joblib.load("model.pkl")

def extract_text_from_email(raw_bytes):
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    text = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text += part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
    else:
        text = msg.get_payload(decode=True).decode("utf-8", errors="replace")
    return text

def classify_text(text):
    label = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    confidence = round(max(proba), 2)
    return label, confidence

# Optie 1️⃣: test met losse tekst
sample_text = "Uw wachtwoord is verlopen, klik hier om te herstellen."
label, conf = classify_text(sample_text)
print(f"Testmail: {label} (confidence {conf})")

# Optie 2️⃣: lees echte e-mails (IMAP)
def fetch_latest_emails(host, username, password, n=5):
    M = imaplib.IMAP4_SSL(host)
    M.login(username, password)
    M.select("INBOX")
    typ, data = M.search(None, "ALL")
    mail_ids = data[0].split()[-n:]
    for i in mail_ids:
        typ, msg_data = M.fetch(i, "(RFC822)")
        raw = msg_data[0][1]
        text = extract_text_from_email(raw)
        label, conf = classify_text(text)
        print("-----")
        print(f"Label: {label} (confidence={conf})")
        print(text[:300], "...")
    M.logout()

# fetch_latest_emails("imap.yourmail.com", "user@example.com", "password")
