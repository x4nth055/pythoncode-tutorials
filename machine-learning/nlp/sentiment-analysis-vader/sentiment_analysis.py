from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# init the sentiment analyzer
sia = SentimentIntensityAnalyzer()

sentences = [
    "This food is amazing and tasty !",
    "Exoplanets are planets outside the solar system",
    "This is sad to see such bad behavior"
]

for sentence in sentences:
    score = sia.polarity_scores(sentence)["compound"]
    print(f'The sentiment value of the sentence :"{sentence}" is : {score}')

for sentence in sentences:
    print(f'For the sentence "{sentence}"')
    polarity = sia.polarity_scores(sentence)
    pos = polarity["pos"]
    neu = polarity["neu"]
    neg = polarity["neg"]
    print(f'The percententage of positive sentiment in :"{sentence}" is : {round(pos*100,2)} %')
    print(f'The percententage of neutral sentiment in :"{sentence}" is : {round(neu*100,2)} %')
    print(f'The percententage of negative sentiment in :"{sentence}" is : {round(neg*100,2)} %')
    print("="*50)