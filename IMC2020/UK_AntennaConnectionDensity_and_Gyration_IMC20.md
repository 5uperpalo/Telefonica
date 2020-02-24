
# Section 1 Initialization
## Section 1.1 Copy results of UK_HomeDetection_IMC20 to HDFS


```bash
%%bash
tar xvzf HomeAntenna_Jan_2020.tar.gz
hdfs dfs -mkdir QoE/Liverpool/Liverpool_Jan_2020/residents
hdfs dfs -mkdir QoE/London/London_Jan_2020/residents
hdfs dfs -mkdir QoE/Birmingham/Birmingham_Jan_2020/residents
hdfs dfs -copyFromLocal HomeAntenna_Jan_2020/Liverpool/part-00000-31b6f85a-4fd4-42fa-b470-ce5b79d473ba-c000.csv QoE/Liverpool/Liverpool_Jan_2020/residents/
hdfs dfs -copyFromLocal HomeAntenna_Jan_2020/London/part-00000-98ea76fa-3237-4fd3-a9c0-4e01e37b7bdf-c000.csv QoE/London/London_Jan_2020/residents/
hdfs dfs -copyFromLocal HomeAntenna_Jan_2020/Birmingham/part-00000-8b70e381-5976-4d29-9431-6fed7cba105a-c000.csv QoE/Birmingham/Birmingham_Jan_2020/residents/
```

## Section 1.2 PySpark and Python modules


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
conf = SparkConf().setAppName('AntennaConnectionDensity')#.setMaster('local[*]')
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
data_dirs = ['QoE/Liverpool/Liverpool_Jan_2020/',
             'QoE/London/London_Jan_2020/',
             'QoE/Birmingham/Birmingham_Jan_2020/']

output_files = ['Liverpool_Jan_2020',
                'London_Jan_2020',
                'Birmingham_Jan_2020']

days = ['01', '02', '03']


gyration_schema = StructType([StructField('device_id', StringType(), True),
                              StructField('gyration', DoubleType(), True),
                              StructField('dt', StringType(), True)])
data_schema = StructType([StructField('device_id', StringType(), True),
                          StructField('antenna_id', StringType(), True),
                          StructField('time_spent', DoubleType(), True),
                          StructField('dt', StringType(), True)])
```

# Section 2 Aggregated User 3/4G Experience and Antenna Connection Density by day
## Section 2.1 Data gathering and transformation

1. iterate over days 
2. split each row on "tab" 
3. create data frames  
  3.1 dataframes have rows where each rows begins with device_id, gyration, 2 mistery values and [antena_id(lkey), time_spent]  pairs   
4. join dataframes with previously identified residents and their home antennas 
5. filter resident/nonresident datasets 
6. group and calculate metrics 


```python
for data_dir,output_file in zip(data_dirs[2:],output_files[2:]):
    home_antenna_df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(data_dir + 'residents/part-*')
    gyration_df = sqlContext.createDataFrame(sc.emptyRDD(), gyration_schema)
    data_df = sqlContext.createDataFrame(sc.emptyRDD(), data_schema)    
    for day in days:
        data_temp = sc.textFile(data_dir + day + '/hour=*/part-*').map(lambda x: x.split('\t'))
        
        gyration_temp_df = data_temp.map(lambda x: [x[i] for i in [0,1]])\
                                    .toDF(('device_id', 'gyration'))\
                                    .withColumn('dt', lit(day))
        gyration_df = gyration_df.unionAll(gyration_temp_df)

        data_temp_df = data_temp.flatMap(lambda x: [(x[0], x[i], x[i+1]) for i in range(5,len(x),2)])\
                                .toDF(('device_id', 'antenna_id', 'time_spent'))\
                                .withColumn('dt', lit(day))
        data_df = data_df.unionAll(data_temp_df)
    
    # it is not possible to rename multiple columns with "withColumnRenamed"
    # the join keeps all columns - even those in join condition -> change column names that are similar
    home_antenna_df = home_antenna_df.withColumnRenamed('device_id', 'device_idd')
    home_antenna_df = home_antenna_df.withColumnRenamed('antenna_id', 'home_antenna_id')

    # 'left' join will create null cells in "home_antenna_id" column for non_residents
    gyration_df = gyration_df.join(home_antenna_df,[home_antenna_df['device_idd']==gyration_df['device_id']], 'left').drop('device_idd')
    
    data_df = data_df.join(home_antenna_df,[home_antenna_df['device_idd']==data_df['device_id'], home_antenna_df['home_antenna_id']==data_df['antenna_id']], 'left').drop('device_idd')
    
    # divide the dataset on resident/nonresidents
    gyration_df = gyration_df.filter(col('home_antenna_id').isNotNull())
    
    data_df_residents = data_df.filter(col('home_antenna_id').isNotNull()).drop('home_antenna_id')
    data_df_nonresidents = data_df.filter(col('home_antenna_id').isNull()).drop('home_antenna_id')

    
    # this step can be easily changed to compute same metrics only for particular days - e.g. weekends/weekdays
    # by adding filter statement to 2nd line : .filter(col('dt').isin(['01','02', ...]))
    # [NOTE] - there can be only 1 groupby statement -> no chaining of grouby, it must be a new command
    gyration_df = gyration_df.groupby('device_id','home_antenna_id','dt')\
                             .agg(sum('gyration').alias('sum_gyration'))
    gyration_df = gyration_df.groupby('device_id','home_antenna_id')\
                             .agg(avg('sum_gyration').alias('avg_sum_gyration'))
    gyration_df.coalesce(1).write.format('com.databricks.spark.csv').option('header', 'true').save(data_dir + 'Gyration_' + output_file + '_residents')
    #gyration_df.coalesce(1).write.format('com.databricks.spark.csv').option('header', 'true').csv(data_dir + 'Gyration_' + output_file + '_residents')
    
    data_df_residents = data_df_residents.groupby('antenna_id','dt')\
                                         .agg(sum('time_spent').alias('sum_time'), count('device_id').alias('count_device_id'))
    data_df_residents = data_df_residents.groupby('antenna_id')\
                                         .agg(avg('sum_time').alias('avg_sum_time'), avg('count_device_id').alias('avg_count_device_id'))
    data_df_residents.coalesce(1).write.format('com.databricks.spark.csv').option('header', 'true').save(data_dir + 'AntennaConnectionDensity_' + output_file + '_residents')
                                         
    data_df_nonresidents = data_df_nonresidents.groupby('antenna_id','dt')\
                                               .agg(sum('time_spent').alias('sum_time'), count('device_id').alias('count_device_id'))
    data_df_nonresidents = data_df_nonresidents.groupby('antenna_id')\
                                               .agg(avg('sum_time').alias('avg_sum_time'), avg('count_device_id').alias('avg_count_device_id'))
    data_df_nonresidents.coalesce(1).write.format('com.databricks.spark.csv').option('header', 'true').save(data_dir + 'AntennaConnectionDensity_' + output_file + '_nonresidents')
```
