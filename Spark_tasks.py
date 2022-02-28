#!/usr/bin/env python
# coding: utf-8

# In[12]:


from pyspark.sql.session import SparkSession
from pyspark.sql import functions as fn

import matplotlib.pyplot as plt


# In[19]:


spark = SparkSession.builder.appName("Analysis").getOrCreate()


# In[10]:


# Reading dataset

vehicles_df = spark.read.csv("hdfs://localhost:9000/mnt/data/source/vehicles.csv", inferSchema=True, header=True)


# In[4]:


# Dropping unused columns

vehicles_df = vehicles_df.drop("_c0")


# In[13]:


vehicles_df.columns


# ## Task-7 Querying Using Spark SQL

# In[6]:


vehicles_df.createOrReplaceTempView("vehicles")


# In[11]:


spark.sql("SELECT * FROM VEHICLES LIMIT 10").toPandas().head()


# In[20]:


# Analysis-1

pd_df = spark.sql("SELECT MANUFACTURER, 100*(COUNT(1)/(SELECT COUNT(1) FROM VEHICLES)) AS PERCENTAGE                     FROM VEHICLES                     GROUP BY MANUFACTURER                     ORDER BY PERCENTAGE DESC                     LIMIT 10").toPandas()

pd_df.plot.pie(y="PERCENTAGE", labels=pd_df["MANUFACTURER"], figsize=(12,12), startangle=145, autopct='%.2f%%')


# In[32]:


# Analysis-2

spark.sql("DROP TABLE IF EXISTS VEHICLES_STG")

spark.sql("CREATE TABLE VEHICLES_STG AS             SELECT CAST(YEAR AS INTEGER) YEAR             FROM VEHICLES")

pd_df = spark.sql("SELECT YEAR, COUNT(1) AS COUNT                     FROM VEHICLES_STG                     WHERE YEAR>=1995 AND YEAR<=2020                     GROUP BY YEAR                     ORDER BY YEAR").toPandas()

pd_df.plot(x="YEAR", y="COUNT", kind="line", figsize=(15, 10))


# ## Task-8 Building Machine Learning Model

# In[46]:


from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoderEstimator, StandardScaler
from pyspark.ml import Pipeline

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# In[94]:


cat_features = ["region", "manufacturer", "model", "condition", "cylinders", "fuel", "transmission",
                "size", "type", "paint_color", "state"]

cat_features_ix = ["region_ix", "manufacturer_ix", "model_ix", "condition_ix", "cylinders_ix", 
                   "fuel_ix", "transmission_ix", "size_ix", "type_ix", "paint_color_ix", "state_ix"]

cat_features_vec = ["region_vec", "manufacturer_vec", "model_vec", "condition_vec", "cylinders_vec", 
                   "fuel_vec", "transmission_vec", "size_vec", "type_vec", "paint_color_vec", "state_vec"]

num_features = ["year", "odometer"]

for col in num_features:
    vehicles_df = vehicles_df.withColumn(col, fn.col(col).cast("Double"))
    
vehicles_df = vehicles_df.withColumn("price", fn.col("price").cast("Double"))

vehicles_df = vehicles_df.dropna()

for col in cat_features:
    indexer = StringIndexer(inputCol=col, outputCol=col+"_ix", handleInvalid="skip")
    vehicles_df = indexer.fit(vehicles_df).transform(vehicles_df)    


# In[95]:


oneHotEncoder = OneHotEncoderEstimator(inputCols=cat_features_ix, outputCols=cat_features_vec, 
                                       handleInvalid="keep")

vehicles_df = oneHotEncoder.fit(vehicles_df).transform(vehicles_df)


# In[96]:


assembler = VectorAssembler(inputCols = cat_features_vec+num_features, 
                            outputCol = "features", handleInvalid="skip")

features_df = assembler.transform(vehicles_df)


# In[97]:


scaler = StandardScaler(inputCol="features", outputCol="sc_features")

scaled_df = scaler.fit(features_df).transform(features_df)


# In[98]:


train, test = scaled_df.randomSplit([0.8, 0.2])

randomForest = RandomForestRegressor()                     .setFeaturesCol("sc_features")                     .setLabelCol("price") 

model = randomForest.fit(train)


# In[100]:


predictions = model.transform(test)

evaluator = RegressionEvaluator()                 .setLabelCol("price")                 .setMetricName("r2")

print("R2 ERROR RATE: {}".format(evaluator.evaluate(predictions)))

