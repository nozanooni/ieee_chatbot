"""
scrape_tweets.py - Scrape @ieee_kau_sb tweets with metadata
Tries multiple Nitter instances. Falls back to existing data if all fail.
"""
import feedparser, json, os, re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

os.makedirs("data/extra_docs", exist_ok=True)

# Try multiple Nitter instances
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.1d4.us",
]
USERNAME = "ieee_kau_sb"

def nitter_to_x(url):
    m = re.search(r'/status/(\d+)', url)
    return f"https://x.com/ieee_kau_sb/status/{m.group(1)}" if m else url

def extract_links(text):
    wa    = re.findall(r'https?://(?:chat\.whatsapp\.com|wa\.me)/\S+', text)
    forms = re.findall(r'https?://(?:forms\.gle|docs\.google\.com/forms)/\S+', text)
    meet  = re.findall(r'https?://meet\.google\.com/\S+', text)
    return {
        "whatsapp": wa[0].rstrip('.,)') if wa else None,
        "form":     forms[0].rstrip('.,)') if forms else None,
        "meet":     meet[0].rstrip('.,)') if meet else None,
    }

def parse_iso(date_raw):
    try:
        dt = parsedate_to_datetime(date_raw)
        return dt.isoformat(), dt
    except:
        return None, None

# Try each Nitter instance
feed = None
for instance in NITTER_INSTANCES:
    url = f"{instance}/{USERNAME}/rss"
    print(f"🐦 Trying {instance}...")
    try:
        f = feedparser.parse(url)
        if f.entries:
            feed = f
            print(f"✅ Success with {instance}")
            break
        else:
            print(f"   ⚠️  No entries")
    except Exception as e:
        print(f"   ❌ Error: {e}")

if not feed or not feed.entries:
    print("⚠️  All Nitter instances failed. Keeping existing data.")
    # Check if we have existing data
    if os.path.exists("data/tweets.json"):
        with open("data/tweets.json") as f:
            existing = json.load(f)
        print(f"   Using existing {len(existing)} tweets from data/tweets.json")
    else:
        print("   No existing data found either.")
    exit(0)

now = datetime.now(timezone.utc)
tweets = []
txt_lines = ["# منشورات نادي IEEE KAU على منصة X\n"]

for entry in feed.entries[:20]:
    title    = entry.get("title", "").strip()
    date_raw = entry.get("published", "")
    iso_date, dt = parse_iso(date_raw)
    x_link   = nitter_to_x(entry.get("link", ""))
    links    = extract_links(title)

    is_upcoming = False
    days_old = None
    if dt:
        diff = (dt - now).total_seconds()
        days_old = -(diff / 86400)
        is_upcoming = diff > -86400  # within last 24h or future

    tweet = {
        "title":      title,
        "date":       iso_date,
        "date_raw":   date_raw,
        "link":       x_link,
        "is_upcoming": is_upcoming,
        "days_old":   round(days_old, 1) if days_old is not None else None,
        "whatsapp":   links["whatsapp"],
        "form":       links["form"],
        "meet":       links["meet"],
    }
    tweets.append(tweet)

    date_label = dt.strftime("%Y-%m-%d %H:%M") if dt else ""
    status = "قادم" if is_upcoming else f"منذ {int(days_old)} يوم" if days_old else ""
    txt_lines.append(f"\n[{date_label}] ({status})")
    txt_lines.append(title)
    if x_link:              txt_lines.append(f"رابط X: {x_link}")
    if links["meet"]:       txt_lines.append(f"رابط الاجتماع: {links['meet']}")
    if links["whatsapp"]:   txt_lines.append(f"رابط واتساب: {links['whatsapp']}")
    txt_lines.append("")

with open("data/tweets.json", "w", encoding="utf-8") as f:
    json.dump(tweets, f, ensure_ascii=False, indent=2)
with open("data/extra_docs/twitter.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(txt_lines))

upcoming = [t for t in tweets if t["is_upcoming"]]
print(f"✅ Saved {len(tweets)} tweets | {len(upcoming)} upcoming")
for u in upcoming:
    d = u['date'][:10] if u['date'] else '?'
    print(f"  📅 {d} — {u['title'][:70]}...")
    if u.get('meet'): print(f"     📹 Meet: {u['meet']}")