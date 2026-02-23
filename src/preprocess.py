import pandas as pd
import re

def clean_text(text):
    """
    Clean text using regex.
    Lowercases text, removes URLs, special characters, and extra spaces.
    """
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_data(fake_path='data/Fake.csv', true_path='data/True.csv'):
    """
    Loads, merges, and preprocesses the fake and true news datasets.
    """

    # Load datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Keep only necessary columns
    fake_df = fake_df[['title', 'text']]
    true_df = true_df[['title', 'text']]

    # Add labels
    fake_df['label'] = 0
    true_df['label'] = 1

    # Merge datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Drop NA only for title/text
    df = df.dropna(subset=['title', 'text'])

    # Ensure string type
    df['title'] = df['title'].astype(str)
    df['text'] = df['text'].astype(str)

    # Combine title + text
    df['content'] = df['title'] + " " + df['text']

    # Clean text
    df['content'] = df['content'].apply(clean_text)

    # Remove duplicates AFTER cleaning
    df = df.drop_duplicates(subset=['content'])

    # Remove short articles
    df['word_count'] = df['content'].apply(lambda x: len(x.split()))
    df = df[df['word_count'] >= 10].copy()

    df = df.drop(columns=['word_count'])

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


if __name__ == '__main__':
    df = preprocess_data()
    print(f"Dataset preprocessed successfully. Shape: {df.shape}")