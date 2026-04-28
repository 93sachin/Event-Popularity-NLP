import pandas as pd
import random
from datetime import datetime, timedelta

event_names = [
    "Music Fest", "Tech Meetup", "Art Workshop", "Business Seminar",
    "Food Festival", "Startup Pitch", "Yoga Session", "Dance Show",
    "Book Fair", "Coding Bootcamp"
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
    "Learn Python and data science"
]

categories = ["Music", "Tech", "Art", "Business", "Food", "Health", "Education"]
locations = ["Delhi", "Mumbai", "Bangalore", "Pune", "Jaipur", "Chandigarh"]

data = []

for i in range(250):  # 🔥 250 rows
    event = random.choice(event_names)
    desc = random.choice(descriptions)
    cat = random.choice(categories)
    loc = random.choice(locations)

    date = datetime.now() + timedelta(days=random.randint(1, 100))
    price = random.randint(0, 1000)
    attendance = random.randint(10, 500)

    # Rule-based popularity (SMART 💀)
    if attendance > 150 or price < 200:
        popularity = 1
    else:
        popularity = 0

    data.append([event, desc, cat, loc, date, price, attendance, popularity])

df = pd.DataFrame(data, columns=[
    "event_name", "description", "category",
    "location", "date_time", "price",
    "past_attendance", "popularity"
])

# Save
df.to_csv("data/events.csv", index=False)

print("✅ Dataset generated with", len(df), "rows")