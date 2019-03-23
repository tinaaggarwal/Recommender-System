from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import monotonically_increasing_id
import math

spark = SparkSession.builder.appName('recommender_system').getOrCreate()

train_set = spark.read.text("train.dat").rdd
line = train_set.map(lambda row: row.value.split("\t"))
line.collect()
training_RDD = line.map(lambda row: Row(UserId=int(row[0]),ItemId=int(row[1]), Rating=int(row[2]), Timestamp=int(row[3])))
training_df = spark.createDataFrame(training_RDD)

test_set = spark.read.text("test.dat").rdd
line_test = test_set.map(lambda row: row.value.split("\t"))
line_test.collect()
testing_RDD = line_test.map(lambda row: Row(UserId=int(row[0]),ItemId=int(row[1])))
testing_df = spark.createDataFrame(testing_RDD)
testing_res = testing_df.withColumn("Index", monotonically_increasing_id())

als = ALS(maxIter=20, regParam=0.14, userCol="UserId", itemCol="ItemId", ratingCol="Rating", coldStartStrategy="nan")
model = als.fit(training_df)
predictions = model.transform(testing_res)

x = predictions.sort(predictions.Index)

pred_col = x.select("prediction").rdd.flatMap(list).collect()

result = []

for pred_val in pred_col:
    if math.isnan(pred_val):
        result.append(2)
    else:
        result.append(int(round(pred_val)))

res_file = open('hw.txt', 'w')
for res in result:
    res_file.write(str(res) + '\n')