import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer


# Load the dataset
df = pd.read_csv('harry_potter_books.csv')

# Combine all text into a single string
all_text = ' '.join(df['text'].values.tolist())

# Define additional stopwords specific to Harry Potter analysis
additional_stopwords = set(['said', 'could', 'would', 'one', 'back', 'like', 'know'])

# Combine standard stopwords with additional stopwords
all_stopwords = STOPWORDS.union(additional_stopwords)

# Generate word cloud excluding stopwords
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=all_stopwords).generate(all_text)

# Plot word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Significant Words in Harry Potter Books')
plt.show()



# # Calculate total length of each chapter
# chapter_lengths = df.groupby(['book', 'chapter'])['text'].apply(lambda x: ' '.join(x)).apply(len).reset_index()
#
# # Plot histogram of chapter lengths
# plt.figure(figsize=(10, 6))
# sns.histplot(chapter_lengths['text'], bins=30, kde=True, color='skyblue')
# plt.title('Distribution of Chapter Lengths in Harry Potter Books')
# plt.xlabel('Chapter Length (characters)')
# plt.ylabel('Density')
# plt.show()
#
# # Calculate total length of each chapter
# chapter_lengths = df.groupby(['book', 'chapter'])['text'].apply(lambda x: ' '.join(x)).apply(len).reset_index()
#
# # Plot histogram of chapter lengths divided by books in a single graph
# plt.figure(figsize=(12, 8))
#
# # Create a palette for different books
# palette = sns.color_palette("husl", len(chapter_lengths['book'].unique()))
#
# # Plot KDEs for each book
# sns.histplot(data=chapter_lengths, x='text', hue='book', multiple='stack', bins=30, palette=palette, kde=True)
#
# # Set plot labels and title
# plt.xlabel('Chapter Length (characters)')
# plt.ylabel('Density')
# plt.title('Distribution of Chapter Lengths by Book in Harry Potter Series')
# plt.legend(title='Book')
# plt.show()





# # Calculate total length of each chapter
# chapter_lengths = df.groupby(['book', 'chapter'])['text'].apply(lambda x: ' '.join(x)).apply(len).reset_index()
#
# # Plot histogram of chapter lengths divided by books in a single graph
# plt.figure(figsize=(12, 8))
#
# # Create a palette for different books
# unique_books = chapter_lengths['book'].unique()
# palette = sns.color_palette("husl", len(unique_books))
#
# # Plot KDEs for each book
# sns.histplot(data=chapter_lengths, x='text', hue='book', multiple='stack', palette=palette, kde=True)
#
# # Set plot labels and title
# plt.xlabel('Chapter Length (characters)')
# plt.ylabel('Frequency (chapters per book)')
# plt.title('Distribution of Chapter Lengths by Book in Harry Potter Series')
#
# # Create custom legend
# handles = [plt.Line2D([0], [0], color=palette[i], lw=4) for i in range(len(unique_books))]
# labels = unique_books
# plt.legend(handles, labels, title='Book', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#
# plt.tight_layout()  # Adjust subplots to fit in figure area.
# plt.show()







# # Combine text of each book
# book_texts = df.groupby('book')['text'].apply(lambda x: ' '.join(x)).reset_index()
#
# # Compute TF-IDF
# vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = vectorizer.fit_transform(book_texts['text'])
# feature_names = vectorizer.get_feature_names_out()
#
# # Convert to DataFrame for easier manipulation
# tfidf_df = pd.DataFrame(tfidf_matrix.T.toarray(), index=feature_names, columns=book_texts['book'])
#
# # Get top N words for each book
# top_n = 10
# top_words = {}
# for book in tfidf_df.columns:
#     sorted_tfidf = tfidf_df[book].sort_values(ascending=False)
#     top_words[book] = sorted_tfidf.head(top_n)
#
# # Create a DataFrame to store the results
# results = []
# for book, scores in top_words.items():
#     for rank, (word, score) in enumerate(scores.items(), 1):
#         results.append((book, word, rank, score))
#
# results_df = pd.DataFrame(results, columns=['book', 'word', 'rank', 'tfidf'])
#
# # Plotting
# fig, axes = plt.subplots(3, 3, figsize=(15, 15))
# axes = axes.flatten()
# palette = sns.color_palette("husl", len(results_df['book'].unique()))
#
# for ax, (book, data) in zip(axes, results_df.groupby('book')):
#     sns.barplot(x='tfidf', y='word', data=data, ax=ax, palette=palette)
#     ax.set_title(book)
#     ax.set_xlabel('TF-IDF Score')
#     ax.set_ylabel('Word')
#     ax.set_xlim(0, results_df['tfidf'].max() * 1.1)
#     for i, (value, name) in enumerate(zip(data['tfidf'], data['word'])):
#         ax.text(value, i, f'{value:.2f}', color='black', ha="left", va="center")
#
# fig.suptitle('Most Significant Words (TF-IDF) per Book', fontsize=20)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()