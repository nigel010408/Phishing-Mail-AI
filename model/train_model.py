"""
rule_based_email_classifier.py

Eenvoudige regelgebaseerde classifier:
- Labels: IMPORTANT, NORMAL, SPAM, PHISHING
- Heuristieken: verdachte woorden, links, mismatched from vs return-path, auth headers, urgency, attachments
- Opslag: SQLite (emails table)
- Voorbeeld: main() bevat test-strings die je kunt uitbreiden
"""

import re
import sqlite3
import email
import imaplib
from email import policy
from email.parser import BytesParser
from urllib.parse import urlparse
from collections import defaultdict
import html
import tldextract

# ---------- Utils ----------
SUSPICIOUS_WORDS = [
    r"wachtwoord", r"account.+(geblokkeerd|geblokkeerd|suspicious|suspendeerd)",
    r"verifi", r"bevestig", r"reset", r"klik hier", r"direct actie", r"betaling vereist",
    r"overschrijving", r"gratis", r"win", r"congratul", r"prij[s|s]en", r"urgent", r"let op"
]
SUSPICIOUS_LINK_PATTERNS = [
    r"bit\.ly", r"tinyurl", r"goo\.gl", r"xn--", r"\.ru/", r"\.cn/", r"\.tk/"
]
SPAM_WORDS = [r"win", r"gratis", r"klik hier", r"aanbieding", r"korting", r"promo", r"unsubscribe"]
WHITELIST_DOMAINS = {"company.nl", "trustedbank.nl"}  # voeg bedrijfsspecifieke domeinen toe
BLACKLISTED_SENDER_DOMAINS = {"badsite.ru", "malicious.example"}

EMAIL_LABELS = ("IMPORTANT", "NORMAL", "SPAM", "PHISHING")

# ---------- Parsing helpers ----------
def extract_text_from_message(msg):
    """Extraheer text (plain of HTML fallback). Retourneer één samengestelde string."""
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                try:
                    parts.append(part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace"))
                except:
                    pass
            elif ctype == "text/html" and not parts:
                # fallback: gebruik HTML als er geen plain text is
                try:
                    html_text = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                    parts.append(strip_html(html_text))
                except:
                    pass
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            try:
                parts.append(payload.decode(msg.get_content_charset() or "utf-8", errors="replace"))
            except:
                parts.append(str(payload))
    return "\n".join(parts).strip()

def strip_html(html_str):
    """Versimpelde HTML -> plain text."""
    text = re.sub(r"<script.*?>.*?</script>", "", html_str, flags=re.S|re.I)
    text = re.sub(r"<style.*?>.*?</style>", "", text, flags=re.S|re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()

def extract_urls(text):
    urls = re.findall(r"https?://[^\s'\"<>]+", text)
    return urls

def get_domain(url):
    try:
        parsed = urlparse(url)
        ext = tldextract.extract(parsed.netloc)
        domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        return domain.lower() if domain else parsed.netloc.lower()
    except:
        return ""

# ---------- Heuristics ----------
def check_suspicious_words(text):
    reasons = []
    score = 0
    lowered = text.lower()
    for pat in SUSPICIOUS_WORDS:
        if re.search(pat, lowered):
            reasons.append(f"Verdacht woord/patroon: {pat}")
            score += 2
    for pat in SPAM_WORDS:
        if re.search(pat, lowered):
            reasons.append(f"Spam woord/patroon: {pat}")
            score += 1
    return score, reasons

def check_links(urls, from_domain):
    reasons = []
    score = 0
    for u in urls:
        dom = get_domain(u)
        # veel shorteners of vreemde TLDs
        if any(re.search(p, u, flags=re.I) for p in SUSPICIOUS_LINK_PATTERNS):
            reasons.append(f"Verdachte shortener/URL: {u}")
            score += 3
        # domein mismatch tussen From en link
        if dom and from_domain and dom != from_domain and dom not in WHITELIST_DOMAINS:
            reasons.append(f"Link naar afwijkend domein: {dom} (link: {u})")
            score += 2
    return score, reasons

def check_sender_headers(msg):
    reasons = []
    score = 0
    from_header = msg.get("From", "")
    return_path = msg.get("Return-Path", "")
    auth_res = msg.get("Authentication-Results", "")
    received_spf = msg.get("Received-SPF", "")

    # eenvoudige heuristiek: mismatch From vs Return-Path
    if return_path and from_header:
        # haal domains
        m_from = re.search(r"@([A-Za-z0-9\.\-]+)", from_header)
        m_ret = re.search(r"@([A-Za-z0-9\.\-]+)", return_path)
        d_from = m_from.group(1).lower() if m_from else ""
        d_ret = m_ret.group(1).lower() if m_ret else ""
        if d_from and d_ret and d_from != d_ret:
            reasons.append(f"From-domain ({d_from}) ≠ Return-Path-domain ({d_ret})")
            score += 2

    # check auth headers (presence van fail of neutral)
    if auth_res and re.search(r"fail|temper|permanent", auth_res, flags=re.I):
        reasons.append("Authentication-Results toont fail/verdacht")
        score += 3
    if received_spf and re.search(r"fail|softfail", received_spf, flags=re.I):
        reasons.append("Received-SPF toont fail/softfail")
        score += 3

    # check blacklisted sender domain
    m = re.search(r"@([A-Za-z0-9\.\-]+)", from_header)
    if m:
        dom = m.group(1).lower()
        if dom in BLACKLISTED_SENDER_DOMAINS:
            reasons.append(f"Afzenderdomein in blacklist: {dom}")
            score += 5
    return score, reasons

def check_attachments(msg):
    reasons = []
    score = 0
    for part in msg.walk():
        if part.get_content_maintype() == 'application' or part.get_filename():
            fname = part.get_filename()
            reasons.append(f"Attachment aanwezig: {fname}")
            # executables of double-extensies zijn extra verdacht
            if fname and re.search(r"\.(exe|scr|bat|cmd|js|vbs)$", fname, flags=re.I):
                reasons.append(f"Zeer verdacht attachment type: {fname}")
                score += 4
            else:
                score += 1
    return score, reasons

# ---------- Core classifier ----------
def classify_email(msg):
    """
    Retourneert dict:
    {
        'label': one of EMAIL_LABELS,
        'score': integer (hoe hoger, hoe verdachter),
        'reasons': [strings],
        'urls': [...],
    }
    """
    text = extract_text_from_message(msg)
    subject = msg.get("Subject", "")
    from_header = msg.get("From", "")
    # domain from 'From'
    m = re.search(r"@([A-Za-z0-9\.\-]+)", from_header or "")
    from_domain = m.group(1).lower() if m else ""

    total_score = 0
    reasons = []

    # 1) suspicious words
    s_score, s_reasons = check_suspicious_words(subject + "\n" + text)
    total_score += s_score
    reasons.extend(s_reasons)

    # 2) links
    urls = extract_urls(subject + "\n" + text)
    l_score, l_reasons = check_links(urls, from_domain)
    total_score += l_score
    reasons.extend(l_reasons)

    # 3) headers
    h_score, h_reasons = check_sender_headers(msg)
    total_score += h_score
    reasons.extend(h_reasons)

    # 4) attachments
    a_score, a_reasons = check_attachments(msg)
    total_score += a_score
    reasons.extend(a_reasons)

    # 5) urgency phrases -> boost score
    if re.search(r"urgent|direct actie|binnen 24 uur|anders sluiten", (subject + " " + text).lower()):
        reasons.append("Urgentie-taal gedetecteerd")
        total_score += 2

    # 6) short message with link only (common phishing)
    if len(text) < 200 and urls and not re.search(r"beste|geachte|met vriendelijke groet", text.lower()):
        reasons.append("Kort bericht met link(s) zonder veel body")
        total_score += 2

    # Heuristische label keuze op basis van score en presence
    label = "NORMAL"
    # thresholds (tweakbaar)
    if total_score >= 8:
        # hoog verdacht -> PHISHING
        label = "PHISHING"
    elif total_score >= 4:
        # verdachte maar mogelijk spam
        label = "SPAM"
    else:
        # check for important keywords
        if re.search(r"factuur|betaling|contract|afspraak|deadline|vergadering", subject + " " + text, flags=re.I):
            label = "IMPORTANT"
        else:
            label = "NORMAL"

    # produce confidence: convert score into 0..1
    confidence = min(1.0, total_score / 10.0)

    return {
        "label": label,
        "score": total_score,
        "confidence": round(confidence, 2),
        "reasons": reasons,
        "urls": urls,
        "subject": subject,
        "from": from_header
    }

# ---------- Storage ----------
def init_db(db_path="emails.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT,
            subject TEXT,
            label TEXT,
            score INTEGER,
            confidence REAL,
            reasons TEXT,
            urls TEXT,
            raw TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def store_result(conn, result, raw):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO emails (sender, subject, label, score, confidence, reasons, urls, raw)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result["from"],
        result["subject"],
        result["label"],
        result["score"],
        result["confidence"],
        "; ".join(result["reasons"]),
        "; ".join(result["urls"]),
        raw
    ))
    conn.commit()

# ---------- IMAP fetch helper (optioneel) ----------
def fetch_mail_from_imap(host, username, password, mailbox="INBOX", limit=10, ssl=True):
    """Voorbeeld: connect en return lijst van email.message.Message"""
    msgs = []
    if ssl:
        M = imaplib.IMAP4_SSL(host)
    else:
        M = imaplib.IMAP4(host)
    M.login(username, password)
    M.select(mailbox)
    typ, data = M.search(None, 'ALL')
    all_ids = data[0].split()
    # laatste N e-mails
    selected = all_ids[-limit:]
    for num in selected:
        typ, msg_data = M.fetch(num, '(RFC822)')
        raw = msg_data[0][1]
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        msgs.append((msg, raw.decode('utf-8', errors='replace')))
    M.logout()
    return msgs

# ---------- Example / Test ----------
def main():
    # Init DB
    conn = init_db(":memory:")  # verander in "emails.db" voor persistente storage

    # Voorbeelden (je kunt ook fetch_mail_from_imap gebruiken)
    raw_examples = [
        # phishing-achtig
        b"From: support@fakebank.com\r\nSubject: Uw rekening is geblokkeerd!\r\n\r\nBeste klant,\r\nUw rekening is geblokkeerd. Klik hier om te herstellen: http://malicious.example/verify\r\nMet vriendelijke groet,\r\nFakeBank",
        # spam
        b"From: promo@somepromo.com\r\nSubject: Win een gratis iPhone!!!\r\n\r\nKlik hier: https://bit.ly/freeiphone\r\nUnsubscribe",
        # important
        b"From: hr@company.nl\r\nSubject: Factuur voor oktober\r\n\r\nBeste,\r\nJe factuur is bijgevoegd. Graag checken.\r\nMet vriendelijke groet,\r\nHR Team",
        # normal
        b"From: collega@company.nl\r\nSubject: Lunch morgen?\r\n\r\nZullen we morgen lunchen om 12:30?"
    ]

    for raw in raw_examples:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        res = classify_email(msg)
        print("----")
        print(f"From: {res['from']}")
        print(f"Subject: {res['subject']}")
        print(f"Label: {res['label']} (score={res['score']}, confidence={res['confidence']})")
        print("Redenen:")
        for r in res['reasons'][:6]:
            print(" -", r)
        print("URLs:", res['urls'])
        # store
        store_result(conn, res, raw.decode('utf-8', errors='replace'))

    # toon opgeslagen rows
    cur = conn.cursor()
    cur.execute("SELECT id, sender, subject, label, score, confidence FROM emails")
    for row in cur.fetchall():
        print("Stored:", row)

if __name__ == "__main__":
    main()
