## About this project

In the Movie Recommendation System using Textual Features project, I developed an intelligent movie recommendation solution that goes beyond traditional genre-based suggestions.
Leveraging natural language processing techniques, the system analyzes textual attributes like genres, keywords, taglines, cast, and directors for a diverse collection of movies. By calculating cosine similarity between movie descriptions, the system suggests movies that share nuanced thematic and contextual connections.
This project showcases my ability to apply machine learning concepts, utilize textual data, and create user-centric solutions that enhance movie enthusiasts' viewing experiences. Moreover, it underscores my proficiency in working with data preprocessing, feature engineering, and algorithmic logic to deliver meaningful and engaging recommendations.

## What is Natural Language Processing?

Natural Language Processing (NLP) is a subfield of artificial intelligence (AI) and linguistics that focuses on the interaction between computers and human languages. NLP aims to enable computers to understand, interpret, and generate human language in a way that is both meaningful and contextually relevant.
The ultimate goal of NLP is to bridge the gap between human communication and computer understanding, allowing machines to process and generate natural language in a manner that is useful and meaningful to humans.

NLP involves a wide range of tasks and techniques, including:

1. **Tokenization**: Breaking down text into individual words, phrases, or symbols (tokens) to analyze their meanings and relationships.

2. **Part-of-Speech Tagging**: Identifying the grammatical parts of speech (e.g., nouns, verbs, adjectives) in a sentence.

3. **Named Entity Recognition**: Identifying entities such as names of people, organizations, locations, dates, and more in text.

4. **Sentiment Analysis**: Determining the emotional tone or sentiment expressed in a piece of text, whether it's positive, negative, or neutral.

5. **Language Modeling**: Building models that predict the likelihood of a sequence of words, often used for text generation and completion.

6. **Machine Translation**: Automatically translating text from one language to another.

7. **Text Classification**: Assigning labels or categories to text documents based on their content.

8. **Information Extraction**: Identifying and extracting specific information from unstructured text.

9. **Question Answering**: Developing systems that can understand and answer questions posed in natural language.

10. **Text Generation**: Creating coherent and contextually relevant text based on a given prompt or input.

NLP techniques often involve the use of machine learning algorithms, such as deep learning models like recurrent neural networks (RNNs) and transformer models like BERT and GPT, to process and understand language patterns. These models are trained on large text datasets to learn the relationships between words, phrases, and sentences, allowing them to perform various NLP tasks.

NLP has numerous applications, including chatbots, virtual assistants, sentiment analysis for social media monitoring, language translation, information retrieval, and content summarization. It plays a crucial role in enabling computers to interact with and understand human language, making it a fundamental technology for tasks that require human-computer communication and interaction.

## What is Cosine Similarity?

Cosine similarity is a metric used to measure the similarity between two vectors in a multi-dimensional space. It is commonly used in various fields, including natural language processing, information retrieval, and recommendation systems. Cosine similarity calculates the cosine of the angle between two vectors, representing how closely the vectors are aligned with each other.

In the context of text data, such as documents, sentences, or in your case, movie descriptions, cosine similarity is often used to quantify the similarity between these texts based on the frequency or presence of words. Here's how it works:

1. **Vector Representation**: Each piece of text is represented as a vector in a high-dimensional space. The dimensions of this space correspond to the words in the text, and the values in the vector represent word frequencies, TF-IDF scores, or other text-based metrics.

2. **Calculating Cosine Similarity**: To measure the similarity between two text vectors A and B, cosine similarity calculates the cosine of the angle between these vectors.

3. **Interpretation**: The resulting cosine similarity value ranges from -1 to 1. A value close to 1 indicates that the vectors are very similar (pointing in almost the same direction), while a value close to -1 indicates they are dissimilar (pointing in opposite directions). A cosine similarity value of 0 suggests that the vectors are orthogonal (perpendicular), implying no similarity.

In the context of your movie recommendation system, cosine similarity is used to quantify the similarity between different movies based on the textual features you've chosen (genres, keywords, tagline, cast, director). Higher cosine similarity values between movies indicate greater textual similarity, suggesting that the movies share common textual attributes and might appeal to similar audiences.


## Now lets break the code down


```python
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

In this section, the necessary libraries are imported. NumPy and pandas are used for data manipulation, difflib is used for finding close matches, and scikit-learn's TfidfVectorizer and cosine_similarity are used for vectorization and calculating cosine similarity, respectively.

```python
movies_data = pd.read_csv('/content/movies.csv')
movies_data.head()
```

Here, a CSV file containing movie data is read into a pandas DataFrame called `movies_data`. The `.head()` function is used to display the first few rows of the DataFrame for a quick overview.

```python
movies_data.shape
```

This line calculates and prints the shape of the DataFrame, which shows the number of rows and columns in the dataset.

```python
selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)
```

A list named `selected_features` is defined, containing the textual features you want to consider for the movie recommendation system. These features include genres, keywords, tagline, cast, and director. The list is then printed to show the selected features.

```python
for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')
```

This loop iterates through the `selected_features` list and fills any missing values in those columns with empty strings.

```python
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
```

A new column named `combined_features` is created by concatenating the selected textual features for each movie with spaces in between. This creates a single string that captures multiple aspects of each movie's information.

```python
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
```

A TF-IDF vectorizer is initialized, and the `fit_transform` function is used to convert the `combined_features` into a matrix of TF-IDF features. TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical representation of text data that takes into account word frequencies and their importance in a collection of documents.

```python
similarity = cosine_similarity(feature_vectors)
```

The `cosine_similarity` function calculates the pairwise cosine similarity between all movie feature vectors in the `feature_vectors` matrix. This results in a similarity matrix where each element `(i, j)` represents the cosine similarity between movie `i` and movie `j`.

```python
movie_name = input(' Enter your favourite movie name : ')
list_of_all_titles = movies_data['title'].tolist()
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
close_match = find_close_match[0]
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
similarity_score = list(enumerate(similarity[index_of_the_movie]))
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
```

In this section, the user is prompted to enter their favorite movie name. The code then finds the closest match to the input movie name in the list of all movie titles. It retrieves the index of the matched movie and calculates the similarity scores of this movie with all other movies using the similarity matrix. These scores are then sorted in descending order to get a list of movies most similar to the input movie.

```python
print('Movies suggested for you : \n')
i = 1
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1
```

This loop iterates through the list of similar movies (`sorted_similar_movies`) and prints the top recommended movies along with their index numbers.

The code then repeats the recommendation process for a new favorite movie entered by the user.

Overall, the code implements a simple movie recommendation system based on textual features using TF-IDF vectorization and cosine similarity. It takes user input for favorite movies and suggests similar movies based on the combined textual information.
