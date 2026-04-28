import pandas as pd
import random
from datetime import datetime, timedelta

event_names = [
    "Music Fest", "Tech Meetup", "Art Workshop", "Business Seminar",
    "Food Festival", "Startup Pitch", "Yoga Session", "Dance Show",
    "Book Fair", "Coding Bootcamp", "AI Conference", "Gaming Event"
]

descriptions = [
    "Live music concert with top artists",
    "AI and ML networking event",
    "Painting workshop for beginners",
    "Business growth strategies and networking",
    "Street food and cultural fest",
    "Pitch your startup to investors",
    "Morning yoga and wellness session",
    "Live dance performances",
    "Book exhibition and author meet",
    "Learn Python and data science",
    "Deep learning and AI conference",
    "Gaming tournament with prizes"
]

categories = ["Music", "Tech", "Art", "Business", "Food", "Health", "Education"]
locations = ["Delhi", "Mumbai", "Bangalore", "Pune", "Jaipur", "Chandigarh"]

data = []

for i in range(5000):  # 🔥 5000 rows
    event = random.choice(event_names)
    desc = random.choice(descriptions)
    cat = random.choice(categories)
    loc = random.choice(locations)

    date = datetime.now() + timedelta(days=random.randint(1, 365))
    price = random.randint(0, 2000)
    attendance = random.randint(10, 2000)

    # 🔥 SMART LOGIC (realistic)
    popularity_score = 0

    if attendance > 800:
        popularity_score += 2
    if price < 300:
        popularity_score += 1
    if "music" in desc.lower() or "festival" in desc.lower():
        popularity_score += 1
    if cat in ["Music", "Tech"]:
        popularity_score += 1

    popularity = 1 if popularity_score >= 2 else 0

    data.append([
        event, desc, cat, loc,
        date, price, attendance, popularity
    ])

df = pd.DataFrame(data, columns=[
    "event_name", "description", "category",
    "location", "date_time", "price",
    "past_attendance", "popularity"
])

df.to_csv("data/events.csv", index=False)

print("🔥 Dataset generated with", len(df), "rows")