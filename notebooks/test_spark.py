from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
builder = (SparkSession.builder
    .appName("smoke")
    .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions","50"))

spark = configure_spark_with_delta_pip(builder).getOrCreate()
print("Spark OK:", spark.version)
spark.range(5).write.mode("overwrite").format("delta").save("data/bronze/_smoke")
print("Delta write OK")
spark.stop()
