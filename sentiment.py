import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Setup
plt.style.use('ggplot')
tqdm.pandas()

# Load Excel data
df = pd.read_csv('Reviews.csv')

print("Original shape:", df.shape)

# Use a smaller subset for analysis
df = df.head(500)
print("Subset shape:", df.shape)

# Plot review score distribution
ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars', figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()

# Tokenize a sample review
example = df['Text'].iloc[50]
print("Sample Review:\n", example)

tokens = nltk.word_tokenize(example)
print("Tokens:", tokens[:10])

tagged = nltk.pos_tag(tokens)
print("POS Tags:", tagged[:10])

entities = nltk.chunk.ne_chunk(tagged)
print("Named Entities:")
entities.pprint()

# VADER Sentiment Analysis
sia = SentimentIntensityAnalyzer()
print("VADER Example:", sia.polarity_scores(example))

# VADER on entire dataset
vader_results = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    vader_results[row['Id']] = sia.polarity_scores(row['Text'])

vader_df = pd.DataFrame(vader_results).T.reset_index().rename(columns={'index': 'Id'})
vader_df = vader_df.merge(df, how='left')

# VADER visualization
ax = sns.barplot(data=vader_df, x='Score', y='compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vader_df, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vader_df, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vader_df, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

# Roberta Sentiment
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(text):
    encoded = tokenizer(text, return_tensors='pt', truncation=True)
    output = model(**encoded)
    scores = softmax(output.logits[0].detach().numpy())
    return {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }

# Combined VADER + Roberta
results = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']

        vader_result = sia.polarity_scores(text)
        vader_result = {f'vader_{k}': v for k, v in vader_result.items()}

        roberta_result = polarity_scores_roberta(text)

        results[myid] = {**vader_result, **roberta_result}
    except RuntimeError:
        print(f'Error at ID: {myid}')

results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

# Compare model scores
sns.pairplot(
    data=results_df,
    vars=[
        'vader_neg', 'vader_neu', 'vader_pos',
        'roberta_neg', 'roberta_neu', 'roberta_pos'
    ],
    hue='Score',
    palette='tab10'
)
plt.show()

# Example comparisons
print("1-Star Review with Highest Roberta Positivity:\n")
print(results_df.query('Score == 1').sort_values('roberta_pos', ascending=False)['Text'].values[0])

print("\n1-Star Review with Highest VADER Positivity:\n")
print(results_df.query('Score == 1').sort_values('vader_pos', ascending=False)['Text'].values[0])

print("\n5-Star Review with Highest Roberta Negativity:\n")
print(results_df.query('Score == 5').sort_values('roberta_neg', ascending=False)['Text'].values[0])

print("\n5-Star Review with Highest VADER Negativity:\n")
print(results_df.query('Score == 5').sort_values('vader_neg', ascending=False)['Text'].values[0])
