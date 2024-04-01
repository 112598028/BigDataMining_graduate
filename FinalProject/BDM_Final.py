# For Spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, trim, explode, desc, col, to_date, year, month, rank, split, when
from pyspark.sql.types import StringType
from functools import reduce
from pyspark.sql.window import Window

# For NLP
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# For Data
import pandas as pd
import numpy as np

#  For Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import missingno as msno
from wordcloud import WordCloud

# For Clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

# For Modeling
from pyspark.ml.feature import Tokenizer, CountVectorizer, VectorAssembler
from pyspark.ml import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score

import re

# For Styling
plt.style.use('default')


#######################
# stopwords with nltk #
#######################
nltk.download('stopwords')
stopwords = stopwords.words('english')
stopwords.append(" ")

#############################
# combine all the csv files #
#############################
spark = SparkSession.builder.appName("BDM Final").getOrCreate()

base_path = "/content/drive/MyDrive/巨量資料/BDM_Final/archive-2/"

file_list = [
    "Covid-19 Twitter Dataset (Apr-Jun 2020).csv",
    "Covid-19 Twitter Dataset (Apr-Jun 2021).csv",
    "Covid-19 Twitter Dataset (Aug-Sep 2020).csv"
]

data_frames = []
for file_name in file_list:
    file_path = base_path + file_name
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    data_frames.append(df)

combined_df = reduce(lambda df1, df2: df1.union(df2), data_frames)

######################
# data preprocessing #
######################
broadcast_stopwords = spark.sparkContext.broadcast(stopwords)

def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)

def preprocess_text(sen):
    # Check if sen is None
    if sen is None:
        return ""

    sentence = sen.lower()

    # Remove HTML tags
    sentence = remove_tags(sentence)

    # Remove punctuation and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    # Split sentence into words
    words = sentence.split()

    # Remove stopwords using the broadcasted list
    filtered_words = [word for word in words if word not in broadcast_stopwords.value]

    # Join words back into a sentence with spaces
    result_sentence = ' '.join(filtered_words)

    return result_sentence

# Define UDFs
remove_tags_udf = udf(remove_tags, StringType())
preprocess_text_udf = udf(preprocess_text, StringType())

# Sample DataFrame
sample_df = combined_df.select("clean_tweet").limit(20)

# Apply UDFs to create a new column
processed_df = sample_df.withColumn("clean_tweet_processed", preprocess_text_udf("clean_tweet"))

# Show the processed DataFrame
processed_df.show(truncate=False)


#######################################
# data preprocessing  with our used_df#
#######################################

spark = SparkSession.builder.appName("BDM_Final").getOrCreate()

# deal with the none situation
clean_tweet_processed_udf = udf(lambda text: preprocess_text(remove_tags(text)) if text else "", StringType())


# preprocessing the clean_tweet column, and rebuild the data frame what we need
used_df = combined_df.withColumn("clean_tweet_processed", clean_tweet_processed_udf("clean_tweet")) \
                    .select("created_at", "favorite_count", "retweet_count", "place", "hashtags", "user_mentions", "clean_tweet_processed", "sentiment")

# show the outcome to check
used_df.show(truncate=False)


df = df['created_at'] = pd.to_datetime(df['created_at'])
df['length'] = df['original_text'].apply(lambda x: len(str(x)) if isinstance(x, (str, float)) and not pd.isna(x) else 0)
df = df.reset_index(drop=True)
df = df[['clean_tweet', 'place', 'created_at', 'sentiment', 'length']]
df.head(2)

######################
# EDA on each columns#
######################

# EDA on "created_at" column
time = df.groupby(['created_at']).size()
monthly = df['created_at'].dt.month.value_counts().sort_index()
plt.figure(figsize=(20,6))
sns.lineplot(x=monthly.index, y = monthly.values, color='green')
plt.title('Monthly distribution of tweets', fontsize=15)

fig = px.line(df,
              x=time.index,
              y=time.values,
              title = 'date of tweets',
              template='simple_white')

fig.update_layout(
    xaxis_title = 'Dates',
    yaxis_title = 'Count of Tweets',
    font=dict(size=17,family="Times New Roman"),)
fig.show()

# EDA on "place" column
location = df['place'].value_counts()[:10]
fig = px.bar(x=location.index,y=location.values,text=location.values,
       color = location.index, color_discrete_sequence=px.colors.sequential.deep,
        title = 'Distribution of Top 10 Locations',
        template = 'simple_white')

fig.update_traces(textposition='inside',
                  textfont_size=11)

fig.update_layout(
    xaxis_title = 'Locations',
    yaxis_title = 'Count of Tweets',
    font=dict(size=17,family="Times New Roman"),)

fig.show()

# EDA on "sentiment" column
sentiment = df["sentiment"].value_counts()
fig = px.pie(values=sentiment.values,
             names=sentiment.index,
             color_discrete_sequence=px.colors.sequential.Greens)
fig.update_traces(textposition='inside',
                  textfont_size=11,
                  textinfo='percent+label')
fig.update_layout(title_text="Category Pie Graph",
                  uniformtext_minsize=12,
                  uniformtext_mode='hide')

fig.show()

# EDA on "favorite_count" column
# filter the favorite_count and clean_tweet_processed
filtered_used_df = used_df.filter(
    (col("favorite_count").cast("float").isNotNull()) &
    (col("clean_tweet_processed") != "") &
    (trim(col("clean_tweet_processed")) != "")
)

# oder by favorite_count
sorted_used_df = filtered_used_df.orderBy(col("favorite_count").cast("float").desc())

# show up top 30
sorted_used_df.show(30, truncate=False)


# EDA on "retweet_count" column
# filter the favorite_count and clean_tweet_processed
retweet_used_df = used_df.filter(
    (col("retweet_count").cast("float").isNotNull()) &
    (col("clean_tweet_processed") != "") &
    (trim(col("clean_tweet_processed")) != "")
)

# oder by favorite_count
retweet_sorted_df = retweet_used_df.orderBy(col("retweet_count").cast("float").desc())

# show up top 30
retweet_sorted_df.show(30, truncate=False)


# EDA on "clean_tweet_processed" column
exploded_df = used_df.select("clean_tweet_processed").selectExpr("explode(split(clean_tweet_processed, ' ')) as word").filter(col("word") != "")

# 計算單詞總頻率
word_count_df = exploded_df.groupBy("word").count().orderBy(desc("count"))

# 顯示結果
word_count_df.show(truncate=False)

# 將 Spark DataFrame 轉換成 Python 字典
word_count_dict = dict(word_count_df.rdd.map(lambda row: (row['word'], row['count'])).collect())

wordcloud = WordCloud(colormap = 'Accent', background_color = 'black').generate_from_frequencies(word_count_dict)

#plot with matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
# plt.savefig('top_30_cloud.png')
plt.show()


# 過濾奇怪的資料
filtered_df = used_df.filter(used_df["created_at"].rlike("^\\d{4}-\\d{2}-\\d{2}$"))

# 將年份萃取出來
filtered_df = filtered_df.withColumn("created_at", to_date("created_at", "yyyy-MM-dd"))
filtered_df = filtered_df.withColumn("year", year("created_at"))

# 將單字拆分成多行，過濾掉空白字詞
exploded_df = filtered_df.select("year", "created_at", "clean_tweet_processed").selectExpr("year", "explode(split(clean_tweet_processed, ' ')) as word", "created_at").filter(col("word") != "")

# 計算每年單詞總頻率
word_count_df = exploded_df.groupBy("year", "word").count().orderBy("year", desc("count"))

# 分別篩選 2020 年和 2021 年的資料
word_count_2020_df = word_count_df.filter(word_count_df["year"] == 2020).select("word", "count").withColumnRenamed("count", "count_2020")
word_count_2021_df = word_count_df.filter(word_count_df["year"] == 2021).select("word", "count").withColumnRenamed("count", "count_2021")

# 顯示結果
print("word count of 2020")
word_count_2020_df.show(truncate=False)

print("word count of 2021")
word_count_2021_df.show(truncate=False)

#增加年、月欄位
filtered_df = used_df.filter(used_df["clean_tweet_processed"].isNotNull())  # 過濾掉空的 clean_tweet_processed

# 過濾奇怪的資料
filtered_df = filtered_df.filter(used_df["created_at"].rlike("^\\d{4}-\\d{2}-\\d{2}$"))

# 將年份萃取出來
filtered_df = filtered_df.withColumn("created_at", to_date("created_at", "yyyy-MM-dd"))
filtered_df = filtered_df.withColumn("year", year("created_at"))
filtered_df = filtered_df.withColumn("month", month("created_at"))

# 使用 explode 將 words 欄位拆分成多行
exploded_df = filtered_df.select("year", "month", "clean_tweet_processed").selectExpr("year", "explode(split(clean_tweet_processed, ' ')) as word", "month").filter(col("word") != "")

# 計算每個年月單字的出現次數
word_count_monthly_df = exploded_df.groupBy("year", "month", "word").count()

window_spec = Window().partitionBy("year", "month").orderBy(desc("count"))

ranked_word_count_df = word_count_monthly_df.withColumn("rank", rank().over(window_spec))
top_words_df = ranked_word_count_df.filter(ranked_word_count_df["rank"] <= 5).select("year", "month", "word", "count")

# 過濾 2020 年的結果
top_words_2020_months_df = top_words_df.filter(col("year") == 2020).select("month", "word", "count")

# 過濾 2021 年的結果
top_words_2021_months_df = top_words_df.filter(col("year") == 2021).select("month", "word", "count")

# 顯示結果
print("Top 5 words in 2020 each months:")
top_words_2020_months_df.show(n=25, truncate=False)

print("Top 5 words in 2021 each months:")
top_words_2021_months_df.show(truncate=False)


# EDA on "hashtags" column
# 過濾掉 hashtags 為 NULL 的資料
filtered_hashtags_df = used_df.filter(used_df["hashtags"].isNotNull())

# 將 hashtags 欄位以逗號分隔，使用 explode 將多個 hashtags 拆分成不同行
exploded_hashtags_df = filtered_hashtags_df.select("hashtags") \
    .withColumn("hashtag", explode(split("hashtags", ",\s*")))

# 計算每個 hashtag 的出現次數
hashtag_count_df = exploded_hashtags_df.groupBy("hashtag").agg(count("*").alias("count"))

# 按次數降序排序
sorted_hashtags_df = hashtag_count_df.orderBy(col("count").desc())

# 顯示結果
print("Most Frequent Hashtags of data")
sorted_hashtags_df.show(truncate=False)

wordcloud_tags = WordCloud(colormap='Accent', background_color='black').generate_from_frequencies(hashtags_count_dict)

# 繪製 WordCloud for hashtags
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud_tags, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
# plt.savefig('top_30_cloud.png')
plt.show()


# 過濾掉 hashtags 為 NULL 的資料
filtered_hashtags_df = used_df.filter(used_df["hashtags"].isNotNull())

# 對 hashtags 欄位進行處理，分割成行數據
hashtags_df = used_df.select("created_at", "hashtags").filter(col("hashtags").isNotNull())
exploded_hashtags_df = hashtags_df.select("created_at", explode(split("hashtags", ",\s*")).alias("hashtag"))

# 過濾奇怪的資料
hashtags_yearly_df = exploded_hashtags_df.filter(exploded_hashtags_df["created_at"].rlike("^\\d{4}-\\d{2}-\\d{2}$"))

# 將年份萃取出來
hashtags_yearly_df = hashtags_yearly_df.withColumn("created_at", to_date("created_at", "yyyy-MM-dd"))
hashtags_yearly_df = hashtags_yearly_df.withColumn("year", year("created_at"))

# 計算每年單詞總頻率
hashtags_year_count_df = hashtags_yearly_df.groupBy("year", "hashtag").count().orderBy("year", desc("count"))

# 分別篩選 2020 年和 2021 年的資料
hashtags_count_2020_df = hashtags_year_count_df.filter(hashtags_year_count_df["year"] == 2020).select("hashtag", "count").withColumnRenamed("count", "count_2020")
hashtags_count_2021_df = hashtags_year_count_df.filter(hashtags_year_count_df["year"] == 2021).select("hashtag", "count").withColumnRenamed("count", "count_2021")

# 顯示結果
print("hashtags count of 2020")
hashtags_count_2020_df.show(truncate=False)

print("hashtags count of 2021")
hashtags_count_2021_df.show(truncate=False)



# 增加年、月欄位
filtered_df = used_df.filter(used_df["hashtags"].isNotNull())  # 過濾掉空的 hashtags

# 過濾奇怪的資料
filtered_df = filtered_df.filter(used_df["created_at"].rlike("^\\d{4}-\\d{2}-\\d{2}$"))

# 將年份萃取出來
filtered_df = filtered_df.withColumn("created_at", to_date("created_at", "yyyy-MM-dd"))
filtered_df = filtered_df.withColumn("year", year("created_at"))
filtered_df = filtered_df.withColumn("month", month("created_at"))

# 使用 explode 將 hashtags 欄位拆分成多行
exploded_hashtags_df = filtered_df.select("year", "month", explode(split("hashtags", ",\s*")).alias("hashtag"))
#exploded_df = filtered_df.select("year", "month", "hashtags").selectExpr("year", "explode(split(hashtags, \",\s*\")) as hashtag", "month").filter(col("hashtag") != "")

# 計算每個年月hashtag的出現次數
tag_count_monthly_df = exploded_hashtags_df.groupBy("year", "month", "hashtag").count()

window_spec = Window().partitionBy("year", "month").orderBy(desc("count"))

ranked_tag_count_df = tag_count_monthly_df.withColumn("rank", rank().over(window_spec))
top_tags_df = ranked_tag_count_df.filter(ranked_word_count_df["rank"] <= 5).select("year", "month", "hashtag", "count")

# 過濾 2020 年的結果
top_tags_2020_months_df = top_tags_df.filter(col("year") == 2020).select("month", "hashtag", "count")

# 過濾 2021 年的結果
top_tags_2021_months_df = top_tags_df.filter(col("year") == 2021).select("month", "hashtag", "count")

# 顯示結果
print("Top 5 hashtags in 2020 each months:")
top_tags_2020_months_df.show(n=25, truncate=False)

print("Top 5 hashtags in 2021 each months:")
top_tags_2021_months_df.show(truncate=False)



# EDA on "user_mentions" column
# 過濾掉 user_mentions 為 NULL 的資料
filtered_mention_df = used_df.filter(used_df["user_mentions"].isNotNull())

# 將 mention 欄位以逗號分隔，使用 explode 將多個 hashtags 拆分成不同行
exploded_mention_df = filtered_mention_df.select("user_mentions") \
    .withColumn("mention", explode(split("user_mentions", ",\s*")))

# 再把資料過濾一次，不要出現數值資料
exploded_mention_df = exploded_mention_df.filter(exploded_mention_df["mention"].rlike("[A-Za-z]"))

# 計算每個 user_mentions 的出現次數
mention_count_df = exploded_mention_df.groupBy("mention").agg(count("*").alias("count"))

# 按次數降序排序
sorted_mention_df = mention_count_df.orderBy(col("count").desc())

# 顯示結果
print("Most Frequent mention of data")
sorted_mention_df.show(truncate=False)


# 將 Spark DataFrame 轉換成 Python 字典
mentions_count_dict = dict(sorted_mention_df.rdd.map(lambda row: (row['mention'], row['count'])).collect())
wordcloud_mentions = WordCloud(colormap='Accent', background_color='black').generate_from_frequencies(mentions_count_dict)

# 繪製 WordCloud for hashtags
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud_mentions, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
# plt.savefig('top_30_cloud.png')
plt.show()



# 過濾掉 hashtags 為 NULL 的資料
filtered_mention_df = used_df.filter(used_df["user_mentions"].isNotNull())

# 將 hashtags 欄位以逗號分隔，使用 explode 將多個 hashtags 拆分成不同行
exploded_mention_df = filtered_mention_df.select("created_at", "user_mentions") \
    .withColumn("mention", explode(split("user_mentions", ",\s*")))

# 再把資料過濾一次，不要出現數值資料
exploded_mention_df = exploded_mention_df.filter(exploded_mention_df["mention"].rlike("[A-Za-z]"))

# 過濾奇怪的資料
mentions_yearly_df = exploded_mention_df.filter(exploded_mention_df["created_at"].rlike("^\\d{4}-\\d{2}-\\d{2}$"))

# 將年份萃取出來
mentions_yearly_df = mentions_yearly_df.withColumn("created_at", to_date("created_at", "yyyy-MM-dd"))
mentions_yearly_df = mentions_yearly_df.withColumn("year", year("created_at"))

# 計算每年單詞總頻率
mentions_year_count_df = mentions_yearly_df.groupBy("year", "mention").count().orderBy("year", desc("count"))

# 分別篩選 2020 年和 2021 年的資料
mentions_count_2020_df = mentions_year_count_df.filter(mentions_year_count_df["year"] == 2020).select("mention", "count").withColumnRenamed("count", "count_2020")
mentions_count_2021_df = mentions_year_count_df.filter(mentions_year_count_df["year"] == 2021).select("mention", "count").withColumnRenamed("count", "count_2021")

# 顯示結果
print("mentions count of 2020")
mentions_count_2020_df.show(truncate=False)

print("mentions count of 2021")
mentions_count_2021_df.show(truncate=False)




# 增加年、月欄位
filtered_df = used_df.filter(used_df["user_mentions"].isNotNull())  # 過濾掉空的 hashtags

# 過濾奇怪的資料
filtered_df = filtered_df.filter(used_df["created_at"].rlike("^\\d{4}-\\d{2}-\\d{2}$"))

# 將年份萃取出來
filtered_df = filtered_df.withColumn("created_at", to_date("created_at", "yyyy-MM-dd"))
filtered_df = filtered_df.withColumn("year", year("created_at"))
filtered_df = filtered_df.withColumn("month", month("created_at"))

# 使用 explode 將 user_mentions 欄位拆分成多行
exploded_mentions_df = filtered_df.select("year", "month", explode(split("user_mentions", ",\s*")).alias("mention"))
#exploded_df = filtered_df.select("year", "month", "hashtags").selectExpr("year", "explode(split(hashtags, \",\s*\")) as hashtag", "month").filter(col("hashtag") != "")

# 計算每個年月mention的出現次數
mention_count_monthly_df = exploded_mentions_df.groupBy("year", "month", "mention").count()
window_spec = Window().partitionBy("year", "month").orderBy(desc("count"))

ranked_mention_count_df = mention_count_monthly_df.withColumn("rank", rank().over(window_spec))
top_mentions_df = ranked_mention_count_df.filter(ranked_mention_count_df["rank"] <= 5).select("year", "month", "mention", "count")

# 過濾 2020 年的結果
top_mentions_2020_months_df = top_mentions_df.filter(col("year") == 2020).select("month", "mention", "count")

# 過濾 2021 年的結果
top_mentions_2021_months_df = top_mentions_df.filter(col("year") == 2021).select("month", "mention", "count")

# 顯示結果
print("Top 5 mentions in 2020 each months:")
top_mentions_2020_months_df.show(n=25, truncate=False)

print("Top 5 mentions in 2021 each months:")
top_mentions_2021_months_df.show(truncate=False)


#######################################
# Text clustering with KMeans and PCA #
#######################################
df['clean_tweet'].fillna('', inplace=True)

# Assuming 'df' is your DataFrame with a 'clean_tweet' column
text_data = df['clean_tweet'].astype(str)

# Combine CountVectorizer and TfidfTransformer for feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(text_data)
print(X.shape)

# kmeans elbow method
distortions = []
K = range(1,10)
for k in K:
    kmean = KMeans(n_clusters=k,random_state=7)
    kmean.fit(X)
    distortions.append(kmean.inertia_)

plt.figure(figsize=(20,5))
plt.plot(K, distortions, '-',color='g')
plt.xlabel('k values')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Use K-means clustering
kmeans = KMeans(n_clusters=9, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# 顯示每個簇的數量
print(df['cluster'].value_counts())



"""


交給你貼了QQ



"""

########################
# Sentiment Prediction #
########################

# 查看 distinct 的 sentiment 值
none_sentiments = combined_df.select("sentiment").distinct()

# 過濾出不是 'pos'、'neu'、'neg' 的資料
none_sentiments_df = none_sentiments.filter((col("sentiment") != "pos") & (col("sentiment") != "neu") & (col("sentiment") != "neg"))

# 顯示表格
none_sentiments_df.show(truncate=False)


# deal with the none situation
clean_tweet_processed_udf = udf(lambda text: preprocess_text(remove_tags(text)) if text else "", StringType())


# preprocessing the clean_tweet column, and rebuild the data frame what we need
model_used_df = combined_df.withColumn("clean_tweet_processed", clean_tweet_processed_udf("clean_tweet")) \
                    .select("created_at", "clean_tweet_processed", "sentiment")

# show the outcome to check
model_used_df.show(truncate=False)

# 過濾掉sentiment為空的狀況
filtered_model_used_df = model_used_df.filter(model_used_df["sentiment"].isNotNull())
filtered_model_used_df.show(truncate=False)


# 選擇 sentiment 欄位中不是 pos, neu, neg 的資料
other_sentiments_df = filtered_model_used_df.filter(~filtered_model_used_df['sentiment'].isin(['pos', 'neu', 'neg']))

# 將 'sentiment' 列轉換為浮點數型態
other_sentiments_df = other_sentiments_df.withColumn('sentiment', other_sentiments_df['sentiment'].cast('float'))

# 從 filtered_model_used_df 中刪除 other_sentiments_df 的資料
filtered_model_used_df = filtered_model_used_df.join(other_sentiments_df, on='sentiment', how='left_anti')

# 對 sentiment 欄位進行轉換
other_sentiments_df = other_sentiments_df.withColumn(
    'sentiment',
    when(other_sentiments_df['sentiment'] > 0.5, 'pos')
    .when(other_sentiments_df['sentiment'] == 0.5, 'neu')
    .otherwise('neg')
)

# 如果需要，將 other_sentiments_df 加回 filtered_model_used_df
filtered_model_used_df = filtered_model_used_df.unionByName(other_sentiments_df)

filtered_model_used_df = filtered_model_used_df.select(
    'created_at',
    'clean_tweet_processed',
    'sentiment',
)

# 顯示other_sentiments_df結果
#other_sentiments_df.show()

# 顯示結果
#filtered_model_used_df.show()

# 定義轉換條件
sentiment_mapping = {
    'pos': 2,
    'neu': 1,
    'neg': 0
}

# 使用 withColumn 進行轉換
transformed_model_used_df = filtered_model_used_df.withColumn(
    'sentiment',
    when(filtered_model_used_df['sentiment'] == 'pos', sentiment_mapping['pos'])
    .when(filtered_model_used_df['sentiment'] == 'neu', sentiment_mapping['neu'])
    .when(filtered_model_used_df['sentiment'] == 'neg', sentiment_mapping['neg'])
)

# 顯示轉換後的 DataFrame
#transformed_model_used_df.show()

# 捨棄 sentiment 欄位為 NULL 的資料
model_final_used_df = transformed_model_used_df.filter(col('sentiment').isNotNull())

# 顯示結果
#model_final_used_df.show()

# 創建 Tokenizer 實例，將文本分割成單詞
tokenizer = Tokenizer(inputCol="clean_tweet_processed", outputCol="words")

# 創建 CountVectorizer 實例，將單詞轉換為詞頻特徵
vectorizer = CountVectorizer(inputCol="words", outputCol="features", vocabSize=1000, minDF=5)

# 建立情緒分析 Pipeline
pipeline = Pipeline(stages=[tokenizer, vectorizer])


# 創建 Tokenizer 實例，將文本分割成單詞
tokenizer = Tokenizer(inputCol="clean_tweet_processed", outputCol="words")

# 創建 CountVectorizer 實例，將單詞轉換為詞頻特徵
vectorizer = CountVectorizer(inputCol="words", outputCol="features", vocabSize=1000, minDF=5)

# 建立 VectorAssembler 實例，將特徵列和標籤列合併為一個特徵向量列
assembler = VectorAssembler(inputCols=["features"], outputCol="feature_vector")

# 建立情緒分析 Pipeline
pipeline = Pipeline(stages=[tokenizer, vectorizer, assembler])

# 使用 Pipeline 擬合和轉換資料
model = pipeline.fit(model_final_used_df)
result = model.transform(model_final_used_df)

# 顯示轉換結果
result.select("created_at", "clean_tweet_processed", "feature_vector", "sentiment").show(truncate=False)

# 获取 features_tensor 和 sentiment_tensor
features_tensor = torch.tensor(result.select("feature_vector").collect(), dtype=torch.float32)
sentiment_tensor = torch.tensor(result.select("sentiment").collect(), dtype=torch.long)

# 合并 features_tensor 和 sentiment_tensor 到一个 TensorDataset
dataset = TensorDataset(features_tensor, sentiment_tensor)

# 定义训练集和测试集的大小
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# 利用 random_split 进行分割
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建 DataLoader
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# CNN
# simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 499, 3)  # 3 is the number of categories

    def forward(self, x):
        # Reshape input
        x = x.view(x.size(0), 1, -1)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# model training
epochs = 10
for epoch in range(epochs):
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        # Add one dimension to the second dimension, in order to meet the requirements of the convolutional layer
        outputs = model(inputs.unsqueeze(1))
        # turn the labels into long type
        loss = criterion(outputs, labels.squeeze().long())
        loss.backward()
        optimizer.step()


model.eval()  # 将模型切换为评估模式

predicted_labels = []
true_labels = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs.unsqueeze(1))
        _, predicted = torch.max(outputs, 1)

        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# 将列表转换为 NumPy 数组
predicted_labels = np.array(predicted_labels)
true_labels = np.array(true_labels)

# 计算准确度
accuracy = accuracy_score(true_labels, predicted_labels)

print(f"Test Accuracy: {accuracy * 100:.2f}%")


#LSTM
# simple LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # need to let the shape of x be (batch_size * sequence_length, input_size)
        x = x.view(x.size(0), -1, x.size(-1))

        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1, :, :])
        return out


# initialize the model, loss function, and optimizer
input_size = 1000
hidden_size = 64
output_size = 3
model = SimpleLSTM(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# model training
epochs = 10
for epoch in range(epochs):
    model.train()

    running_loss = 0.0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze().long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # print the training loss
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_dataloader)}")


    model.eval()  # 將模型切換為評估模式

lstm_predicted_labels = []
lstm_true_labels = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        print(inputs.shape)  # 添加此行

        # 確保輸入的形狀是 (batch_size, sequence_length, input_size)
        inputs = inputs.unsqueeze(1)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        lstm_predicted_labels.extend(predicted.cpu().numpy())
        lstm_true_labels.extend(labels.cpu().numpy())

# 將列表轉換為 NumPy 數組
lstm_predicted_labels = np.array(lstm_predicted_labels)
lstm_true_labels = np.array(lstm_true_labels)

# 计算准确度
lstm_accuracy = accuracy_score(lstm_true_labels, lstm_predicted_labels)
print(f"LSTM Test Accuracy: {lstm_accuracy * 100:.2f}%")


# CNN+LSTM
class CombinedModel(nn.Module):
    def __init__(self, cnn_input_size, lstm_input_size, hidden_size, output_size):
        super(CombinedModel, self).__init__()

        # CNN layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc_cnn = nn.Linear(64 * cnn_input_size, lstm_input_size)

        # LSTM layer
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, batch_first=True)

        # Fully connected layer for sentiment prediction
        self.fc_final = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # CNN layer
        x_cnn = x.view(x.size(0), 1, -1)
        x_cnn = self.pool(F.relu(self.conv1(x_cnn)))
        x_cnn = x_cnn.view(x_cnn.size(0), -1)
        x_cnn = F.relu(self.fc_cnn(x_cnn))

        # LSTM layer
        x_lstm, _ = self.lstm(x_cnn.unsqueeze(1))

        # Fully connected layer for sentiment prediction
        x_final = self.fc_final(x_lstm[:, -1, :])

        return x_final

# Initialize the model, criterion, and optimizer
cnn_input_size = 499  # Change this based on the output size of your CNN layer
lstm_input_size = 64  # Change this based on the hidden size of your LSTM layer
hidden_size = 64      # Change this based on your requirements
output_size = 3       # Number of sentiment classes

combined_model = CombinedModel(cnn_input_size, lstm_input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(combined_model.parameters(), lr=0.001)

# Train the combined model
epochs = 10
for epoch in range(epochs):
    # Set the model to training mode
    combined_model.train()

    running_loss = 0.0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = combined_model(inputs)
        loss = criterion(outputs, labels.squeeze().long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the training loss
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_dataloader)}")


combined_model.eval()  # 將模型切換為評估模式

com_predicted_labels = []
com_true_labels = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        # print(inputs.shape)  # 添加此行

        # 確保輸入的形狀是 (batch_size, sequence_length, input_size)
        inputs = inputs.unsqueeze(1)

        outputs = combined_model(inputs)
        _, predicted = torch.max(outputs, 1)

        com_predicted_labels.extend(predicted.cpu().numpy())
        com_true_labels.extend(labels.cpu().numpy())

# 將列表轉換為 NumPy 數組
com_predicted_labels = np.array(com_predicted_labels)
com_true_labels = np.array(com_true_labels)

# count accuracy
COM_accuracy = accuracy_score(com_true_labels, com_predicted_labels)
print(f"Combined Model Test Accuracy: {COM_accuracy * 100:.2f}%")