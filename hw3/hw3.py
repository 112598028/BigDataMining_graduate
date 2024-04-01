from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, udf, explode, concat_ws
from pyspark.ml.feature import Tokenizer, MinHashLSH
from pyspark.sql.types import StringType, ArrayType, IntegerType
from pyspark.ml.linalg import Vectors, VectorUDT 
import re
import os

# Create a Spark session
spark = SparkSession.builder.appName("hw3").getOrCreate()

# User defines the appropriate path to the files' folder
file_path = "./data"

"""
Data Preprocessing
"""

# Get the list of file names
files = spark.sparkContext.wholeTextFiles(f"{file_path}/*.sgm")

# Initialize an empty dataframe (df) that will host the documents
df = spark.createDataFrame([("",)], ["Article"])

# Recursive visit to path, loading of the next sgm file, and appropriate splitting into articles
# until all sgm files are processed and appended to df
for file_path, content in files.collect():
    temp = content.replace("\n", " ")
    temp = re.split("</REUTERS>", temp)
    temp = spark.createDataFrame([(t,) for t in temp], ["Article"])
    df = df.union(temp)

# Remove empty rows from df
df = df.filter(col("Article") != " ").dropDuplicates()

# Confirm correct loading of files: 21578 rows (each row represents one article)
# df.show(truncate=False)

# Add a new column, namely ID, that will contain a unique identifier per article
# OLDID is used for this purpose
df = df.withColumn("ID", udf(lambda x: re.sub(".*OLDID=\"(.*?)\".*", "\\1", x), StringType())("Article"))

# Add a new column, namely BODY, that will contain the main text per article in lowercase
# Articles with no BODY tag are removed (2535 in total)
df = df.filter(col("Article").contains("<BODY>"))
df = df.withColumn("BODY", udf(lambda x: re.sub(".*<body>(.*?)<\/body>.*", "\\1", x.lower()), StringType())("Article"))

# Remove the first column containing full article details (we do not need it anymore)
df = df.drop("Article")

# Replace everything that is not a letter (punctuation, numbers, other symbols, etc.) with white space
df = df.withColumn("BODY", udf(lambda x: re.sub("[^a-zA-Z\\s]", " ", x), StringType())("BODY"))

# Shrink down to just one white space
df = df.withColumn("BODY", udf(lambda x: re.sub("[\\s]+", " ", x), StringType())("BODY"))

# Splitting documents into words
# Create two new columns from df:
# Column s contains the BODY of each document split into its words
# Column id contains the ID of each document
df = df.withColumn("s", udf(lambda x: re.split(" ", x), StringType())("BODY"))
df = df.drop("BODY").withColumnRenamed("s", "BODY")

# Remove records with less than 8 words (24 documents in total)
# so that we can create up to 7-word shingles
df = df.filter(udf(lambda x: len(x) >= 8, StringType())("BODY").cast("boolean"))

# Show the resulting dataframe
# df.show(truncate=False)



"""
Shingling
"""
# Assuming 'df' is your DataFrame with 'ID' and 'BODY' columns
# Replace 'df' with your actual DataFrame name

# Tokenize the 'BODY' column
tokenizer = Tokenizer(inputCol="BODY", outputCol="tokenized_words")
df = tokenizer.transform(df)

# Extract unique shingles sets across all documents
doc_dict = sorted(df.select("tokenized_words").rdd.flatMap(lambda x: x[0]).distinct().collect())

# Define a UDF to convert the tokenized words to a list of integers
list_to_int_array_udf = udf(lambda lst: [1 if shingle in lst else 0 for shingle in doc_dict], ArrayType(IntegerType()))

# Apply the UDF to create the "Characteristic" matrix
Char_Mat = df.withColumn("features", list_to_int_array_udf("tokenized_words")).select("ID", "features")

print(doc_dict)
Char_Mat.show(truncate=False)

# Specify the output directory (not just the file name)
Char_Mat_output_path = "out_csv"

#  Write the DataFrame to a CSV file in overwrite mode
Char_Mat = Char_Mat.withColumn("features_str", concat_ws(",", "features"))

# delete .DS_Store file
ds_store_path = os.path.join(Char_Mat_output_path, ".DS_Store")
if os.path.exists(ds_store_path):
    os.remove(ds_store_path)

Char_Mat.select("ID", "features_str").write.csv(Char_Mat_output_path, header=True, mode="overwrite", sep=",", quoteAll=True)

print(f"成功將 Char_Mat 表格輸出成 CSV 檔案，檔案路徑為: {Char_Mat_output_path}")

"""
Min-Hashing signatures
"""
# Assuming you have a DataFrame Char_Mat with a column "features" containing dense vectors

# Create a UDF to convert dense vectors to VectorUDT
dense_vector_to_vector_udf = udf(lambda v: Vectors.dense(v), VectorUDT())

# Apply the UDF to create a new column "minhash_signature"
Char_Mat = Char_Mat.withColumn("minhash_signature", dense_vector_to_vector_udf("features"))

# Number of hash functions (H)
num_hash_functions = 10  # You can adjust this number based on your requirements

# Create the MinHashLSH model
minhash_lsh = MinHashLSH(inputCol="minhash_signature", outputCol="hashValues", numHashTables=num_hash_functions)

# Fit the model to the data
model = minhash_lsh.fit(Char_Mat)

# Optionally, you can persist the model for later use
# model.save("path/to/save/model")

# Use the model to transform the DataFrame and get hash values
transformed_df = model.transform(Char_Mat)

# Show the resulting DataFrame
transformed_df.select("id", "hashValues").show(truncate=False)

# # Concatenate the columns into a single column
# transformed_df = transformed_df.withColumn(
#     "combined_column", concat_ws(",", col("id"), col("hashValues_str"))
# )

# # Specify the path where you want to save the results
# MinHashLSH_output_path = "out2.txt"

# # Save the DataFrame to a text file
# transformed_df.select("combined_column").write.text(MinHashLSH_output_path)


"""
Locality-Sensitive Hashing: candidate pairs
"""

# # Create MinHashLSH model
# minhash_lsh = MinHashLSH(inputCol="minhash_signature", outputCol="hashValues", numHashTables=5)
# model = minhash_lsh.fit(transformed_df)

selected_df = transformed_df.select("id", "hashValues").limit(2000)

# Explode the hashValues column to get a separate row for each hash value
exploded_df = selected_df.select("id", explode("hashValues").alias("hash")).repartition(100, "hash")

# Group by the hash value and count the number of occurrences
hash_counts = exploded_df.sample(fraction=0.1).groupBy("hash").count()

# Filter for hash values that occur more than once (candidate pairs)
#candidate_pairs = hash_counts.filter("count > 1")
candidate_pairs = hash_counts.filter(hash_counts["count"] > 1)

# Show the candidate pairs
candidate_pairs.show()

spark.stop()



