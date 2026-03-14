import feedparser
import os

os.makedirs("data/extra_docs", exist_ok=True)

rss_url = "https://nitter.net/ieee_kau_sb/rss"
feed = feedparser.parse(rss_url)

tweets = []

for entry in feed.entries[:10]:
    tweets.append(
        f"""
Event Post
Title: {entry.title}
Date: {entry.published}
Link: {entry.link}
"""
    )

with open("data/extra_docs/twitter.txt", "w", encoding="utf-8") as f:
    for t in tweets:
        f.write(t + "\n")

print("Saved tweets:", len(tweets))