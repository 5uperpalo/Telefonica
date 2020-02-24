
# Section 1 Initialization 
## Section 1.1 PySpark and Python modules


```python
import os
os.environ["SPARK_HOME"] = '/usr/local/spark/spark-1.6.2-bin-hadoop2.6'
os.environ['PYSPARK_SUBMIT_ARGS'] = "--master local[*] --deploy-mode client --packages com.databricks:spark-csv_2.11:1.3.0 pyspark-shell"

import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark import sql
from pyspark.sql.types import DoubleType, StructType, StructField, StringType
from pyspark.sql import HiveContext
from pyspark.sql.functions import col, lit, count, sum, avg

print('starting')
conf = SparkConf().setAppName('HomeDetection')#.setMaster('local[*]')
sc = SparkContext(conf=conf)

# we need HiveContext to use Hive builtin functions:
# hive builtin functions : https://support.treasuredata.com/hc/en-us/articles/360001457367-Hive-Built-in-Aggregate-Functions
sqlContext = HiveContext(sc)

print('finished')
```

    starting
    finished


## Section 1.2 Define datasets we are going to use


```python
NightHours = ['/hour=00-04/part-*', '/hour=04-08/part-*']

data_dirs = ['QoE/Liverpool/Liverpool_Jan_2020/',
             'QoE/London/London_Jan_2020/',
             'QoE/Birmingham/Birmingham_Jan_2020/']

output_files = ['HomeAntenna_Liverpool_Jan_2020.csv',
               'HomeAntenna_London_Jan_2020.csv',
               'HomeAntenna_Birmingham_Jan_2020.csv']

days = ['01', '02', '03']

schema = StructType([StructField('device_id', StringType(), True),
                     StructField('antenna_id', StringType(), True),
                     StructField('time_spent', DoubleType(), True)])
```

# Section 2 Home antenna detection

1. iterate over days 
2. split each row on "tab" 
3. create data frames from two 4hour night intervals  
  3.1 dataframes have rows where each rows begins with device_id, gyration, 2 mistery values and [antena_id(lkey), time_spent]  pairs  
  3.2 each device_id has multiple rows 
4. unite both time intervals, sum times for same antennas, keep the antenna with max time_spent 
5. append dataframes from each day to final dataframe 


```python
for data_dir,output_file in zip(data_dirs,output_files):
    NightHours_00040008_final_df = sqlContext.createDataFrame(sc.emptyRDD(), schema)
    for day in days:
        NightHours_0004 = sc.textFile(data_dir + day + NightHours[0]).map(lambda x: x.split('\t'))
        NightHours_0008 = sc.textFile(data_dir + day + NightHours[1]).map(lambda x: x.split('\t'))
        NightHours_0004_df = NightHours_0004.filter(lambda x: float(x[1]) <= 2000).flatMap(lambda x: [(x[0], x[i], x[i+1]) for i in range(5,len(x),2)])\
                                      .toDF(('device_id', 'antenna_id', 'time_spent'))
        
        NightHours_0008_df = NightHours_0008.filter(lambda x: float(x[1]) <= 2000).flatMap(lambda x: [(x[0], x[i], x[i+1]) for i in range(5,len(x),2)])\
                                      .toDF(('device_id', 'antenna_id', 'time_spent'))
        
        w = Window.partitionBy('device_id')
        NightHours_00040008_df = NightHours_0004_df.unionAll(NightHours_0008_df)\
                                      .groupby(['device_id', 'antenna_id']).agg(sum('time_spent').alias('time_spent'))\
                                      .withColumn('max_time_spent', F.max('time_spent').over(w))\
                                      .where(F.col('time_spent') == F.col('max_time_spent'))\
                                      .drop('max_time_spent')
        
        NightHours_00040008_final_df = NightHours_00040008_final_df.unionAll(NightHours_00040008_df)
    NightHours_00040008_final_df = NightHours_00040008_final_df.groupby(['device_id', 'antenna_id']).agg(count('antenna_id').alias('count_antenna_id'))\
                                                               .filter(col('count_antenna_id')>13)\
                                                               .drop('count_antenna_id')
    NightHours_00040008_final_df.toPandas().to_csv(output_file, index=False)
```
