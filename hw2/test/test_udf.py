import pytest
from chispa import assert_df_equality
from pyspark.sql import SparkSession

import pyspark.sql.functions as F

from chispa.column_comparer import assert_column_equality

from video_analytics.functions import median_udf, split_udf


@pytest.fixture(scope='session')
def spark():
    return SparkSession.builder \
        .master("local") \
        .appName("chispa") \
        .getOrCreate()

def test_udf_median_even(spark):

    data = [
        (2, 10),
        (2, 1),
        (2, 4),
        (2, 6)
    ]
    df = spark.createDataFrame(data, ["id", "value"])
    actual_df = df.groupBy('id').agg(median_udf(df['value']))


    expected_data = [
        (2, 5.0),
    ]
    expected_df = spark.createDataFrame(expected_data, ["id", "median_udf(value)"])

    assert_df_equality(actual_df, expected_df)

def test_udf_median_odd(spark):

    data = [
        (2, 10),
        (2, 1),
        (2, 4),
    ]
    df = spark.createDataFrame(data, ["id", "value"])
    actual_df = df.groupBy('id').agg(median_udf(df['value']))


    expected_data = [
        (2, 4.0),
    ]
    expected_df = spark.createDataFrame(expected_data, ["id", "median_udf(value)"])

    assert_df_equality(actual_df, expected_df)

def test_udf_split(spark):

    data = [
        (1, "hello|world"),
        (2, "a|b|c|d"),
    ]
    df = spark.createDataFrame(data, ["id", "text"])
    actual_df = df.withColumn("splitted", split_udf(df['text']))

    expected_data = [
        (1, "hello|world", ["hello", "world"]),
        (2, "a|b|c|d", ["a", "b", "c", "d"]),
    ]
    expected_df = spark.createDataFrame(expected_data, ["id", "text", "splitted"])

    assert_df_equality(actual_df, expected_df)
