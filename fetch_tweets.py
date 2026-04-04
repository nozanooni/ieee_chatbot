import snscrape.modules.twitter as sntwitter
import json

USERNAME = "ieee_kau_sb"

tweets = []

print("Fetching tweets...")

for tweet in sntwitter.TwitterUserScraper(USERNAME).get_items():

    tweets.append({
        "title": tweet.content[:120],
        "description": tweet.content,
        "link": tweet.url,
        "date": str(tweet.date)
    })

    if len(tweets) >= 10:
        break

with open("data/events.json", "w", encoding="utf-8") as f:
    json.dump(tweets, f, ensure_ascii=False, indent=2)

print("Saved", len(tweets), "tweets")