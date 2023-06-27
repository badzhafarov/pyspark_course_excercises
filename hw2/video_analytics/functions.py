from pyspark.sql import SparkSession
from pyspark.sql.pandas.functions import pandas_udf, PandasUDFType

spark = SparkSession.builder.enableHiveSupport().getOrCreate()

@pandas_udf("double", PandasUDFType.GROUPED_AGG)
def median_udf(v):
    return v.median()


@pandas_udf("array<string>", PandasUDFType.SCALAR)
def split_udf(v):
    return v.str.split("|")




