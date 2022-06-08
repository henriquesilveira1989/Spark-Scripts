from pyspark.sql.session import SparkSession
import pyspark.sql.functions as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TABLE_NAME = "dim_people"
TRUSTED_DATASET = f"smartfit-ai-project:analytics_smartfit_trusted.{TABLE_NAME.upper()}"

FILENAME = f"{TABLE_NAME}.parquet"
GCS_INPUT_RAW_DIR = "gs://analytics-smartfit/analytics-data/raw-data"
INPUT_FULL_FILENAME = f"{GCS_INPUT_RAW_DIR}/{TABLE_NAME}/{FILENAME}"

if __name__ == "__main__":
	spark = SparkSession \
		.builder \
		.getOrCreate()

	# Reading from Raw Data
	logger.info(f"Fetching Raw data from Google Cloud Storage: {INPUT_FULL_FILENAME}")

	DDLSchema = 'ID INTEGER, GENDER STRING, BIRTHDAY DATE, LOAD_DATETIME timestamp'

	df = (spark.read.schema(DDLSchema).parquet(INPUT_FULL_FILENAME)
		  .select(F.col("ID").alias("PERSON_ID"),
				  F.when(F.col("GENDER") == "M", 1).otherwise(0).alias("GENDER"),
				  "BIRTHDAY",
				  F.col("LOAD_DATETIME").alias("SF_LOAD_DATETIME"),
				  F.expr("year(current_date()) - year(BIRTHDAY) as AGE"))
		  .where("PERSON_ID IS NOT NULL"))

	# Drop Duplicated:
	df = df.dropDuplicates(subset=["PERSON_ID"])

	# Calculate Median for ages between 14 and 85
	df_age_14_85 = df.filter("AGE>=14 and AGE<=85")
	mediana = df_age_14_85.approxQuantile("AGE", [0.5], 0)[0]

	# Replacing ages > 14 or > 85 by the median
	df = (df.withColumn("new_age", F.when(F.col("AGE").between(14, 85), F.col("AGE")).otherwise(int(mediana)))
		  .drop("AGE")
		  .withColumnRenamed("new_age", "AGE"))

	# Adding BQ_LOAD_DATETIME and dropping duplicates
	df_transformed = df.withColumn("BQ_LOAD_DATETIME", F.current_timestamp()).drop_duplicates()

	# Writing to Big Query
	temporary_bucket = "analytics-smartfit/analytics-data/bigquery-temp"
	spark.conf.set('temporaryGcsBucket', temporary_bucket)
	df_transformed.write.format('bigquery').option('table', TRUSTED_DATASET).save(mode="overwrite")
	logger.info(f"Trusted data saved successfully in Big Query: {TRUSTED_DATASET}")
