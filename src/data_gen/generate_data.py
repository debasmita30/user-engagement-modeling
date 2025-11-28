import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta

np.random.seed(42)

# ------------------------------------------
# Generate Users
# ------------------------------------------
def generate_users(n_users=5000):
    users = []
    start_date = datetime(2025, 1, 1)

    for i in range(n_users):
        user_id = f"user_{i}"
        signup_date = start_date + timedelta(days=np.random.randint(0, 60))
        user_tier = np.random.choice(["free", "premium"], p=[0.8, 0.2])
        org_size = np.random.choice(["small", "medium", "large"], p=[0.6, 0.3, 0.1])

        users.append([user_id, signup_date, user_tier, org_size])

    df_users = pd.DataFrame(users, columns=["user_id", "signup_date", "user_tier", "org_size"])
    return df_users


# ------------------------------------------
# Generate Events
# ------------------------------------------
def generate_events(users_df, days=45, avg_events_per_day=8):
    events = []
    start = datetime(2025, 8, 1)

    event_types = [
        ("sign_in", 0.05),
        ("message_sent", 0.50),
        ("meeting_join", 0.12),
        ("screen_share", 0.08),
        ("file_upload", 0.06),
        ("reaction", 0.15),
        ("sign_out", 0.03),
        ("error", 0.01),
    ]

    for user_id in users_df["user_id"]:
        for d in range(days):
            current_date = start + timedelta(days=d)
            n = np.random.poisson(avg_events_per_day)

            for _ in range(n):
                ev = np.random.choice([e[0] for e in event_types], p=[e[1] for e in event_types])
                ts = current_date + timedelta(seconds=np.random.randint(0, 86400))

                events.append([
                    str(uuid.uuid4()),
                    user_id,
                    ts,
                    ev,
                    np.random.choice(["desktop", "mobile", "web"]),
                    np.random.choice(["US", "IN", "EU", "APAC"]),
                    None,  # session_id (we fill after SQL stitching)
                    np.random.exponential(300) if ev == "meeting_join" else None,
                    np.random.normal(120, 40),
                    int(np.random.rand() < 0.004)
                ])

    df_events = pd.DataFrame(events, columns=[
        "event_id", "user_id", "ts", "event_type", "device_type",
        "region", "session_id", "duration_sec", "latency_ms", "crash"
    ])

    return df_events


# ------------------------------------------
# MAIN
# ------------------------------------------
if __name__ == "__main__":
    print("Generating users...")
    users_df = generate_users()
    users_df.to_csv("data/users.csv", index=False)

    print("Generating events...")
    events_df = generate_events(users_df)
    events_df.to_csv("data/events.csv", index=False)

    print("Data generation complete → check data/ folder.")
