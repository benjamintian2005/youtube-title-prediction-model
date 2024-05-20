from googleapiclient.discovery import build
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np

load_dotenv()

api_key = os.getenv('YOUTUBE_API_KEY')
youtube = build('youtube', 'v3', developerKey=api_key)

words = ["the", "what", "you","this", "that", "with", "in", "be", "and", "of", "to","have", "and", "on","is", "for", "it", "on", "win" , "I" , "ever"]
data = []

for word in words:
    request = youtube.search().list(
        part="snippet",
        maxResults=50,
        order = "relevance",
        q= word, # you can adjust the query to get different sets of data
        regionCode = "US"
    )
    response = request.execute()

    video_ids = [item['id']['videoId'] for item in response['items'] if 'videoId' in item['id']]

    video_details = youtube.videos().list(
        part="snippet,statistics",
        id=','.join(video_ids)
    ).execute()


    for video in video_details['items']:
        title = video['snippet']['title']
        view_count = int(video['statistics']['viewCount'])
        data.append((title, view_count))
#train the data using linear regression
titles, view_counts = zip(*data)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(titles).toarray()
y = np.array(view_counts)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

test = "skibidi rizz "
X = vectorizer.transform([test]).toarray()

print(int(model.predict(X)))

