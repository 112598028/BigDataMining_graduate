import pyspark
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql import Row,SQLContext
from pyspark.sql.functions import col,desc,substring,lit,udf,length
from pyspark.sql.functions import split, explode, monotonically_increasing_id, substring_index

from pyspark.sql.types import StructType,StructField,IntegerType,DoubleType,StringType
import pyspark.sql.functions as f

import csv
from operator import add
import pandas as pd

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords = stopwords.words('english')
stopwords.append("")

# Create a Spark session
spark = SparkSession.builder.appName("WordCount").getOrCreate() 
## spark = SparkSession.builder.appName("WordCount").master("spark://input_your_master_ip").getOrCreate()

path = "./Data/" 
## path = "put_your_file_address"


# Read the news data
data = spark.read.option("escape", '"').csv( path + "News_Final.csv", header=True)

# select the column and cast the data type
data = data.withColumn('SentimentTitle', data['SentimentTitle'].cast('double'))
data = data.withColumn('SentimentHeadline', data['SentimentHeadline'].cast('double'))
data = data.withColumn('PublishDate', data['PublishDate'].cast('date'))
data = data.withColumn('PublishDate', data['PublishDate'].cast('string'))

# data processing for the missing value
data.filter(data.Headline.isNull()).count()
data.filter(data.Source.isNull()).count()
data = data.na.fill("missing")


# ==========
# Question 1
# ==========

# select the column
news_title_total = data.select('Title')
news_headline_total = data.select('Headline')

# clean the useless word
def lower_clean_str(x):
    punc='!"#”$%&\'()*+—–,./:;<=>?@[\\]^_’‘`{|}~-…'
    #punc = '''!()-—–[]{};:”'"\[\\]{|}, <>.…/?+@#$%^&*_~\n=’‘'''
    lowercased_str = x.lower()
    for ch in punc:
        lowercased_str = lowercased_str.replace(ch, '')
    return lowercased_str

# claculate the words frequency
def handle(df_news_total):
    rdd_lines = df_news_total.rdd.map(lambda r: lower_clean_str(r[0]))
    rdd_news_counts = rdd_lines.flatMap(lambda x: x.split(' ')) \
                      .map(lambda x: (x, 1)) \
                      .reduceByKey(add).filter(lambda x: x[0] not in stopwords).toDF(("word", "total"))
    rdd_news_counts = rdd_news_counts.orderBy(desc("total"), "word")
    return rdd_news_counts

rdd_news_title_total = handle(news_title_total)
rdd_news_head_total = handle(news_headline_total)

print("Title total")
rdd_news_title_total.show()
print("Headline total")
rdd_news_head_total.show()

# select the topic name 
topic_list = ['economy','microsoft','obama','palestine']
lower_clean_str_udf = udf(lower_clean_str)

#lower word and remove symbol
data = data.withColumn('Title',lower_clean_str_udf(data["Title"]))
data = data.withColumn('Headline',lower_clean_str_udf(data["Headline"]))

df_news_cate = data.select('Title','Headline', 'Topic').where(data["Topic"].isin(topic_list))
df_news_date = data.select('Title','Headline', 'PublishDate').where((data['PublishDate']!="missing")&(data["Topic"].isin(topic_list)))

df_news_title_total_cate = df_news_cate.withColumn('Total_Title_Word',f.explode(f.split(f.column('Title'), ' ')))

# filter stopwords
df_news_title_total_cate = df_news_title_total_cate.filter(~df_news_title_total_cate["Total_Title_Word"].isin(stopwords))\
                        .groupBy('Topic' ,'Total_Title_Word')\
                        .count()\
                        .sort(['Topic','count'],ascending=[True,False])
df_news_head_total_cate = df_news_cate.withColumn('Total_Headline_Word',f.explode(f.split(f.column('Headline'), ' ')))
# filter stopwords
df_news_head_total_cate = df_news_head_total_cate.filter(~df_news_head_total_cate["Total_Headline_Word"].isin(stopwords))\
                        .groupBy('Topic' ,'Total_Headline_Word')\
                        .count()\
                        .sort(['Topic','count'],ascending=[True,False])

# print the outcome
for topic in topic_list:
    topic_df = df_news_title_total_cate.filter(data['Topic'] == topic)
    topic_df.show(truncate=False)

for topic in topic_list:
    topic_df = df_news_head_total_cate.filter(data['Topic'] == topic)
    topic_df.show(truncate=False)



df_news_title_total_date = df_news_date.withColumn('Total_Title_day',f.explode(f.split(f.column('Title'), ' ')))
#filter stopwords
df_news_title_total_date = df_news_title_total_date.filter(~df_news_title_total_date["Total_Title_day"].isin(stopwords))\
                        .groupBy('PublishDate' ,'Total_Title_day')\
                        .count()\
                        .sort(['PublishDate','count'],ascending=[True,False])
df_news_head_total_date = df_news_date.withColumn('Total_Headline_day',f.explode(f.split(f.column('Headline'), ' ')))
#filter stopwords
df_news_head_total_date = df_news_head_total_date.filter(~df_news_head_total_date["Total_Headline_day"].isin(stopwords))\
                        .groupBy('PublishDate' ,'Total_Headline_day')\
                        .count()\
                        .sort(['PublishDate','count'],ascending=[True,False])

print("Total number of words in Title Sort By Date")
df_news_title_total_date.show()
print("Total number of words in Headlines Sort By Date")
df_news_head_total_date.show()



# ==========
# Question 2
# ==========
platform_list = ["Facebook","GooglePlus","LinkedIn"]

for platform in platform_list:
  # Read Social Feedback Data
  # Assuming the file name is social_feedback.csv, and columns are 'IDLink', 'TS1', 'TS2', ..., 'TS144'
  social_feedback_data = spark.read.csv(path + platform +"_*.csv",header='true', inferSchema='true')

  # Initialize an empty DataFrame to store the results
  hour_result_df = None

  # Create columns for each different hour
  for i in range(1, 145, 3):
      start_index = i
      end_index = i + 2
      hour_name = f"hour{i // 3 + 1}"

      # Calculate the sum for each hour
      hour_sum = col(f"TS{start_index}") + col(f"TS{start_index + 1}") + col(f"TS{end_index}")

      # Calculate the average and add the result to the result DataFrame
      if hour_result_df is None:
          hour_result_df = social_feedback_data.withColumn(hour_name, hour_sum / 3).select("IDLink", hour_name)
      else:
          hour_result_df = hour_result_df.join(
              social_feedback_data.withColumn(hour_name, hour_sum / 3).select("IDLink", hour_name), "IDLink", "outer"
          )

  day_result_df = None
  # Create columns for each different day
  for day_start_index in range(1, 145, 72):
      day_end_index = day_start_index + 71
      day_name = f"day{(day_start_index + 71) // 72}"

      # Calculate the sum for each day
      day_sum = sum(col(f"TS{i}") for i in range(day_start_index, day_end_index + 1))

      # Calculate the average and add the result to the result DataFrame
      if day_result_df is None:
          day_result_df = social_feedback_data.withColumn(day_name, day_sum / 72).select("IDLink", day_name)
      else:
          day_result_df = day_result_df.join(
              social_feedback_data.withColumn(day_name, day_sum / 72).select("IDLink", day_name), "IDLink", "outer"
          )

  # save to csv
  hour_result_df.toPandas().to_csv( platform + "_popular_hour.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
  day_result_df.toPandas().to_csv( platform + "_popular_day.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)


  # Show the table
  print( platform + " the average popularity by hour" )
  hour_result_df.show()

  print( platform + " the average popularity by day" )
  day_result_df.show()




# ==========
# Question 3
# ==========

def handle_senti(type1,type2):
    df_news_senti = data.where(data["Topic"].isin(topic_list)).select('Topic',type1,type2)
    tmp = df_news_senti.groupBy("Topic").sum(type2).orderBy("Topic").toPandas()
    df_seni_topic = list(tmp["Topic"])
    df_seni_sum = list(tmp["sum("+type2+")"])
    df_seni_avg = list(df_news_senti.groupBy("Topic").avg(type2).orderBy("Topic").toPandas()['avg('+type2+')'])
    pdf_news_senti_out = pd.DataFrame(zip(df_seni_topic,df_seni_sum,df_seni_avg),columns=['topic','sum','average'])

    pdf_news_senti_out = spark.createDataFrame(pdf_news_senti_out)
    print(type2)
    pdf_news_senti_out.show()

handle_senti("Title","SentimentTitle")
handle_senti("Headline","SentimentHeadline")


# ==========
# Question 4
# ==========

# Matrix size
list_top100 = []

def remove_others(string):
    global list_top100
    lowercased_str = string.lower()
    punc='!"#”$%&\'()*+—–./:;<=>?@[\\]^_’ ‘`{|}~-…'
    for ch in punc:
        lowercased_str = lowercased_str.replace(ch, ',')
    tsets = lowercased_str.split(',')
    alist = [x for x in tsets if x in list_top100]
    if not alist:
        alist = ["No Values"]
    return ','.join(alist)

def handle_type_co_occurrence(df_word_count,df_news_type,topic,type1):
    global list_top100
    list_top100 = list(df_word_count.select('Total_'+type1+'_Word').toPandas()['Total_'+type1+'_Word'][0:100])

    remove_others_udf = udf(remove_others)

    df_news_type_new = df_news_type.withColumn("New_sentence", remove_others_udf(df_news_type[type1])).select("New_sentence").where(col('New_sentence') != "No Values")
    
    tmp_df = (df_news_type_new.withColumn("id", monotonically_increasing_id()).select("id", f.explode(f.split("New_sentence", ","))))
    df_occurrence_matrix = tmp_df.withColumnRenamed("col", "col_").join(tmp_df, ["id"]).stat.crosstab("col_", "col")

    print("Matrix: "+topic+" "+type1)
    df_occurrence_matrix.show()


for topic in topic_list:
    print(topic)
    tmp_df_title = df_news_cate.select('Title') .where(col('Topic') == topic)
    tmp_df_head = df_news_cate.select('Headline').where(col('Topic') == topic)
    handle_type_co_occurrence(df_news_title_total_cate.where(df_news_title_total_cate["Topic"] == topic),tmp_df_title,topic,'Title')
    handle_type_co_occurrence(df_news_head_total_cate.where(df_news_head_total_cate["Topic"] == topic),tmp_df_head,topic,'Headline')


spark.stop()