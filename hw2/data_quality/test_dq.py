import re

import pyspark.sql.functions as F
import pytest
from pyspark.pandas import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import BooleanType, StructField, StringType, IntegerType, StructType
from soda.scan import Scan

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

comments_schema = StructType([
    StructField("video_id", StringType(), True),
    StructField("comment_text", StringType(), True),
    StructField("likes", IntegerType(), True),
    StructField("replies", IntegerType(), True)])


@pytest.fixture(scope='session')
def spark():
    return SparkSession.builder \
        .master("local") \
        .appName("chispa") \
        .getOrCreate()


def build_scan(name, spark_session):
    scan = Scan()
    scan.disable_telemetry()
    scan.set_scan_definition_name("data_quality_test")
    scan.set_data_source_name("spark_df")
    scan.add_spark_session(spark_session)
    return scan


def test_comment_text(spark: SparkSession, path: str = '../hw1/datasets/UScomments.csv',
                      threshold=0.05):
    comments = spark.read.option('header', 'true').schema(comments_schema) \
        .option("mode", "PERMISSIVE").option("columnNameOfCorruptRecord", "corrupt_record").csv(path)

    def check_text_comment(s):
        try:
            return re.sub('\W+', '', emoji_pattern.sub(r'', s)).replace(" ", "").isalnum()
        except:
            return False

    def check_corrupted_comments(df: DataFrame, column: str = "comment_text"):

        df.cache()

        corrupted_ratio = 1 - (df.na.drop().count() / df.count())
        if corrupted_ratio > threshold:
            raise Exception(f"check data, corrupted_ratio {corrupted_ratio}")

        df = df.withColumn("is_correct", convertUDF(F.col(column)))
        df = df.select("is_correct").cache()

        length = df.count()
        corrupted = df.filter(F.col("is_correct") == False).count()
        corrupted_ratio = corrupted / length

        if corrupted_ratio > threshold:
            raise Exception(f"check data, corrupted_ratio {corrupted_ratio}")

    convertUDF = F.udf(lambda z: check_text_comment(z), BooleanType())

    check_corrupted_comments(comments)


def test_data_correctness(spark: SparkSession, path: str = '../hw1/datasets/UScomments.csv',
                          threshold_video_id: float = 0.01, threshold_video_id_likes_replies: float = 0.9):
    comments = spark.read.option('header', 'true').option("mode", "DROPMALFORMED").schema(comments_schema).csv(
        path
    )

    # 1 check schema

    assert comments_schema == comments.schema

    # 2 check video_id
    correct_length = 11

    comments = comments.withColumn('len', F.length(F.col("video_id"))).cache()
    count = comments.count()
    count_incorrect = comments.filter(F.col('len') != correct_length).count()
    assert threshold_video_id > count_incorrect / count

    # 3 check likes and replies
    comments = comments.withColumn("likes_and_replies", F.col("likes") + F.col("replies"))
    check_likes_and_replies = (F.col("likes_and_replies") > 0) & (F.col("replies") >= 0)
    count_incorrect_counters = comments.filter(~check_likes_and_replies).count()
    assert threshold_video_id_likes_replies > count_incorrect_counters / count
