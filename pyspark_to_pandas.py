# save a file to pandas from pyspark dataframe in case of little memory

from pyspark.sql improt SparkSession, SQLContext
from pyspark.context import SparkContext, SparkConf
from pyspark.sql.types import StringType, BooleanType, IntegerType, StructType
from pyspark.sql import functions as functs
from pyspark.sql.functions import broadcast, concat, col, lit, isnan, trim
from pyspark import StorageLevel

import os, sys
import pandas as pd
#suppose you have a pyspark dataframe  called df_agg

#delete if the folder where we are going to save already exists
delete_folder = "hdfs dfs -rm -r tobesaved"
os.system(delete_folder)
#same for local
delete_folder2 = "rm -rf tobesaved_local"
os.system(delete_folder2)

#save df_agg in memory disk and heap
df_agg.persist(storageLevel=StorageLevel(True, True, True, False,1))
#divide the pyspark dataframe in X number of partitions e.g. 1000
df_agg.coalesce(1000).write.format("com.databricks.spark.csv").option("header","true").save("tobesaved")
#once it's saved copy to local
local_cmd = "hdfs dfs -get tobesaved tobesaved_local"
os.system(local_cmd)
#unpersist
df_agg.unpersist()
list_of_files = os.listdir("tobesaved_local")
list_of_files.remove("_SUCCESS")
all_dfs = []
for filename in list_of_files:
    path_file = "tobesaved_local/" + filename
    df = pd.read_csv(path_file)
    all_dfs.append(df)
#open up
pandas_dataframe = pd.concat(all_dfs, ignore_index=True)
#end :) 
