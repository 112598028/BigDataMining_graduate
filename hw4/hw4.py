from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
import time
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.feature import VectorAssembler,  MinMaxScaler
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType



spark = SparkSession.builder.appName("HW4").getOrCreate()
sc=spark.sparkContext

file_path = "./ml-1m/"
out_path = "./output/"

################
###### Q1 ######
################

## 先讀進 rating.dat 的資料
text_file = sc.textFile(file_path + "ratings.dat")
start = time.time()

# data 以 "::" 分隔
def deal_process(data):
    data = data.map(lambda x : x.split('::'))
    return data

data = deal_process(text_file)
data = data.collect()
rating_df = spark.createDataFrame(data).toDF('UserID','MovieID','Rating','Timestamp')

text_file = sc.textFile(file_path+"users.dat")

data = deal_process(text_file)
data = data.collect()
user_df =  spark.createDataFrame(data).toDF('UserID','Gender','Age','Occupation','Zip-code')

df_union = rating_df.join(user_df, on='UserID', how='inner')

df_union = df_union.withColumn('Rating', df_union['Rating'].cast('double'))

avg = df_union.select('MovieID','Rating')
avg_movie = avg.groupBy('MovieID').mean().sort('avg(Rating)',ascending=False)
avg_movie.show()

################
###### Q2 ######
################

grouped_list = ['Gender', 'Age', 'Occupation', ]

for group in grouped_list:
    avg_df = df_union.select('MovieID', 'Rating', group)
    avg_df = avg_df.groupBy('MovieID', group).mean().sort('avg(Rating)',ascending=False)
    avg_df.show()

################
###### Q3 ######
################

text_file = sc.textFile(file_path + "movies.dat")

data = deal_process(text_file)
data = data.collect()
# ID, title, 類型
movie_df =  spark.createDataFrame(data).toDF('MovieID','Title','Genres')
# movie_df.show()

df_union = df_union.join(movie_df, on='MovieID', how='inner')
# df_union.show()

avg = df_union.select('UserID','Rating')
avg_rate = avg.groupBy('UserID').mean().sort('avg(Rating)',ascending=False)
avg_rate.show()

# avg.toPandas().to_csv(out_path + "q3_user.csv",index=False)

avg = df_union.select('UserID','Rating','Genres')
avg_genres = avg.groupBy('UserID','Genres').mean().sort('avg(Rating)',ascending=False)
avg_genres.show()


################
###### Q4 ######
################

# 轉換 'avg' DataFrame 中的 'avg(Rating)' 列為 'features' 列
assembler = VectorAssembler(inputCols=['avg(Rating)'], outputCol='features')
user_vectors = assembler.transform(avg_rate).select('UserID', 'features')

# 正規化 features 列
scaler = MinMaxScaler(inputCol='features', outputCol='scaled_features')
scaler_model = scaler.fit(user_vectors)
scaled_user_vectors = scaler_model.transform(user_vectors).select('UserID', 'scaled_features')

# 計算用戶相似性分數
cosine_similarity_udf = F.udf(lambda x, y: float(x.dot(y)), FloatType())

similarity_matrix = scaled_user_vectors.alias('u1') \
    .join(scaled_user_vectors.alias('u2'), F.col('u1.UserID') != F.col('u2.UserID')) \
    .select(
        F.col('u1.UserID').alias('User1'),
        F.col('u2.UserID').alias('User2'),
        cosine_similarity_udf('u1.scaled_features', 'u2.scaled_features').alias('CosineSimilarity')
    )

# 以排序後的 users 為基準，計算最高相似性分數
window_spec = Window.partitionBy('User1').orderBy(F.desc('CosineSimilarity'))
ranked_similarity = similarity_matrix.withColumn('rank', F.row_number().over(window_spec)) \
    .filter('rank = 1') \
    .select(F.array(F.col('User1'), F.col('User2')).alias('UserPair'), 'CosineSimilarity') \
    .orderBy(F.desc('CosineSimilarity'))

# 顯示結果
ranked_similarity.show(truncate=False)

################
###### Q5 ######
################

# 轉換 'avg' DataFrame 中的 'avg(Rating)' 列為 'features' 列
assembler = VectorAssembler(inputCols=['avg(Rating)'], outputCol='features')
movie_vectors = assembler.transform(avg_movie).select('MovieID', 'features')

# 正規化 features 列
scaler = MinMaxScaler(inputCol='features', outputCol='scaled_features')
scaler_model = scaler.fit(movie_vectors)
scaled_movie_vectors = scaler_model.transform(movie_vectors).select('MovieID', 'scaled_features')

# 計算電影相似性分數
cosine_similarity_udf = F.udf(lambda x, y: float(x.dot(y)), FloatType())

similarity_matrix = scaled_movie_vectors.alias('m1') \
    .join(scaled_movie_vectors.alias('m2'), F.col('m1.MovieID') != F.col('m2.MovieID')) \
    .select(
        F.col('m1.MovieID').alias('Movie1'),
        F.col('m2.MovieID').alias('Movie2'),
        cosine_similarity_udf('m1.scaled_features', 'm2.scaled_features').alias('CosineSimilarity')
    )

# 以排序後的 movies 為基準，計算最高相似性分數
window_spec = Window.partitionBy('Movie1').orderBy(F.desc('CosineSimilarity'))
ranked_similarity = similarity_matrix.withColumn('rank', F.row_number().over(window_spec)) \
    .filter('rank = 1') \
    .select(F.array(F.col('Movie1'), F.col('Movie2')).alias('MoviePair'), 'CosineSimilarity') \
    .orderBy(F.desc('CosineSimilarity'))

# 顯示結果
ranked_similarity.show(truncate=False)

################
###### Q6 ######
################

# Convert the DataFrame to an RDD of Row objects
ratingsRDD = df_union.rdd.map(lambda row: Row(
    userId=int(row['UserID']),
    movieId=int(row['MovieID']),
    rating=float(row['Rating']),
    timestamp=int(row['Timestamp'])
))

ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)

# Generate top 10 movie recommendations for a specified set of users
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
# Generate top 10 user recommendations for a specified set of movies
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)

print("Top 10 movie recommendations for each user:")
userRecs.show(truncate=False)

print("Top 10 user recommendations for each movie:")
movieRecs.show(truncate=False)

print("Top 10 movie recommendations for specified users:")
userSubsetRecs.show(truncate=False)

print("Top 10 user recommendations for specified movies:")
movieSubSetRecs.show(truncate=False)


spark.stop()