import pandas as pd
import os
import random

def create_large_test_data():
    # Configuration
    NUM_ROWS = 1000
    FILENAME = "large_test_sentiment.csv"

    # Templates to generate random text
    # We mix these to ensure the model sees different sentiments
    positive_templates = [
        "I absolutely love this product! It works perfectly.",
        "Highly recommended! Five stars.",
        "The user interface is intuitive and easy to use.",
        "Great value for money, I will buy again.",
        "Customer service was amazing and very helpful.",
        "This is exactly what I was looking for.",
        "Incredible performance and fast shipping.",
        "The quality exceeded my expectations.",
        "A fantastic experience from start to finish.",
        "Simply the best item in its category."
    ]

    negative_templates = [
        "This is the worst service I have ever received. Terrible.",
        "Do not buy this, it broke after one day.",
        "Customer support was rude and unhelpful.",
        "Complete waste of money, very disappointed.",
        "The app crashes constantly, unusable.",
        "Shipping was delayed by two weeks, unacceptable.",
        "Quality is very poor compared to the description.",
        "I regret purchasing this immediately.",
        "Does not work as advertised. Beware.",
        "The worst experience I've had in a long time."
    ]

    neutral_templates = [
        "It's okay, nothing special but does the job.",
        "I'm not sure how I feel about this yet.",
        "The delivery was late but the item quality is good.",
        "Just an average experience, honestly.",
        "It works, but there are better alternatives.",
        "Decent for the price, but don't expect miracles.",
        "Mixed feelings about this purchase.",
        "It is what it is. Neither good nor bad.",
        "Standard quality, nothing to write home about.",
        "It arrived today, haven't tested it fully yet."
    ]

    all_templates = positive_templates + negative_templates + neutral_templates

    # Generate Data
    print(f"Generating {NUM_ROWS} rows of random data...")
    
    data = {
        'id': range(1, NUM_ROWS + 1),
        'text': [random.choice(all_templates) for _ in range(NUM_ROWS)],
        'metadata': [f'batch_run_{random.randint(1, 5)}' for _ in range(NUM_ROWS)]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(FILENAME, index=False)
    
    print(f"✅ Success! Created '{FILENAME}' with {len(df)} records.")
    print(f"📍 Location: {os.path.abspath(FILENAME)}")
    print("------------------------------------------------")
    print("Columns found:", list(df.columns))
    print("First 5 rows preview:")
    print(df.head())

if __name__ == "__main__":
    create_large_test_data()