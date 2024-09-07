import numpy as np
import yfinance as yf
import pandas as pd
from newsapi.newsapi_client import NewsApiClient
import datetime as dt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
import boto3
from io import StringIO

# Load Nifty500 Historical Data using yfinance
nifty500 = yf.Ticker("^CRSLDX")

# Define the date range for historical data
start_date = dt.date.today() - dt.timedelta(days=365)  # Fetch 1 year of data
end_date = dt.date.today()

# Fetch historical data (adjust the period and interval as needed)
nifty500_hist = nifty500.history(start=start_date, end=end_date)

# Process the historical data (example: calculate returns or any feature)
nifty500_hist['Daily Return'] = nifty500_hist['Close'].pct_change() * 100  # Daily percentage change

# News API Data remains the same
api = NewsApiClient(api_key='xxxxxxxxxxxxxxxxxxxxxxxx')

today = dt.datetime.today().strftime('%Y-%m-%d')
yesterday = (dt.datetime.today() - dt.timedelta(1)).strftime('%Y-%m-%d')

news = api.get_everything(domains='moneycontrol.com',
                          language='en',
                          to=today,
                          from_param=yesterday,
                          page=1,
                          page_size=100)

news = news['articles']

# Sentiment Analysis using VADER
analyzer = SentimentIntensityAnalyzer()
news_content = [article['title'] for article in news]

sentiment_scores = [analyzer.polarity_scores(content) for content in news_content]

# Convert sentiment analysis results to DataFrame
sentiment_df = pd.DataFrame(sentiment_scores)
sentiment_df['Date'] = pd.to_datetime([article['publishedAt'] for article in news])

# Merge with historical stock data (assuming you want to align by date)
# Adjust according to the data you want to join
nifty500_hist['Date'] = nifty500_hist.index.date
final_data = pd.merge(nifty500_hist, sentiment_df, on='Date', how='inner')

# Load the pre-trained model and make predictions
resource = boto3.resource('s3',
                          aws_access_key_id='xxxxxxxxxxxxxxxxxxxxx',
                          aws_secret_access_key='xxxxxxxxxxxxxxxxxxxxxxxxxxxx',
                          region_name='us-east-1')
model = pickle.loads(resource.Bucket('myniftyperformance').Object('finalized_model.sav').get()['Body'].read())

# Predict based on final data
predictions = model.predict(final_data)

# Logic to interpret predictions remains the same
for pred in predictions:
    if pred == 0:
        print('The market will be flat')
    elif pred == 1:
        print('The market will be negative')
    elif pred == 2:
        print('The market will be positive')

# Save results back to S3 as a CSV
client = boto3.client('s3',
                      aws_access_key_id='xxxxxxxxxxxxxxxxxxxxxx',
                      aws_secret_access_key='xxxxxxxxxxxxxxxxxxxxxxxxxxx',
                      region_name='us-east-1')

# Get the existing CSV file from S3
obj = client.get_object(Bucket='myniftyperformance', Key='theperformancedata.csv')
data = pd.read_csv(obj['Body'])

# Concatenate the new data with the existing data
updated_file = pd.concat([data, final_data])

# Save the updated file back to S3
csv_buf = StringIO()
updated_file.to_csv(csv_buf, header=True, index=False)
csv_buf.seek(0)
client.put_object(Bucket='myniftyperformance', Body=csv_buf.getvalue(), Key='theperformancedata.csv')

print('Success')

