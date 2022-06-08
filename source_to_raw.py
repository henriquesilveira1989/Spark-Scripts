from pyspark.sql.session import SparkSession
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TABLE_NAME = "dim_people"
SCHEMA = "dw"
PARTITION_COLUMN = "ID"  # column to use for partition

# Output Location
FILENAME = f"{TABLE_NAME}.parquet"
GCS_RAW_DIR = "gs://analytics-teste/analytics-data/raw-data"
FULL_FILENAME = f"{GCS_RAW_DIR}/{TABLE_NAME}/{FILENAME}"


def get_connections():
	"""
	Seta a conexão para o AWS-REDSHIFT
	"""
	aws_redshift_user = spark.conf.get("spark.executorEnv.AWS_REDSHIFT_USER")
	aws_redshift_pass = spark.conf.get("spark.executorEnv.AWS_REDSHIFT_PASS")
	aws_redshift_host = spark.conf.get("spark.executorEnv.AWS_REDSHIFT_HOST")
	aws_redshift_port = spark.conf.get("spark.executorEnv.AWS_REDSHIFT_PORT", "5439")
	aws_redshift_database = spark.conf.get("spark.executorEnv.AWS_REDSHIFT_DATABASE", "data")
	driver = "com.amazon.redshift.jdbc42.Driver"
	jdbc_url = f"jdbc:redshift://{aws_redshift_host}:{aws_redshift_port}/{aws_redshift_database}"
	logger.info(f"JDBC url:\n {jdbc_url}")
	return {'url': jdbc_url, 'driver': driver, 'user': aws_redshift_user, 'password': aws_redshift_pass}


def get_boundaries(conn, boundary_query):
	"""
	Função tem como objetivo executar uma query inicial no banco de dados para buscar
	os limites mínimo e máximo pela primary key. Uma vez com esses limites será possível particionar e distribuir de
	maneira mais equitativa o processamento paralelo entre os workers no cluster.

	:param conn: parâmetros de conexão
	:param boundary_query: query a ser realizada para identificar os limites

	:return: objeto contendo valor min (lowerBound) e max (upperBound)
	"""
	return (spark.read.format("jdbc")
			.option("query", boundary_query)
			.option("driver", conn["driver"])
			.option("url", conn["url"])
			.option("user", conn["user"])
			.option("password", conn["password"])
			).load().collect()[0]


def get_full_load_queries():
	"""
	Função para retornar as queries de Full Load.

	:return:
		bound_query: query para buscar os limites (lowerBound e upperBound).
		query: query para fazer a busca full no Redshift.
	"""
	# Getting boundaries to enable parallelism
	full_bound_query = f"select min({PARTITION_COLUMN}) as min, max({PARTITION_COLUMN}) as max from {SCHEMA}.{TABLE_NAME}"
	# Full Load Query
	full_query = f"(select * from {SCHEMA}.{TABLE_NAME}) as dbtable"
	return full_bound_query, full_query


def main():
	"""
	Função principal, chamada iniciada pelo spark-submit (DataProcPySparkOperator).
		1. Seta o particionamento dos dados;
		2. Define as conexões com o Redshift;
		3. Define as queries para load full ou load incremental;
		4. Executa a "bound query" para buscar os limites de particionamento;
		5. Executa a query no Redshift;
		6. Salva o resultado em parquet no Google Cloud Storage.
	"""

	# getting default number of cpus
	numPartitions = spark.sparkContext.defaultParallelism
	# Configure default shuffle partitions to match the number of cores:
	spark.conf.set("spark.sql.shuffle.partitions", numPartitions)
	# Let's use four times the number of partitions per core
	numPartitions = numPartitions * 4

	# Connection Details
	connections = get_connections()
	(bound_query, query) = get_full_load_queries()

	# Getting boundaries to enable parallelism
	logger.info(f"Fetching boundaries:\n {bound_query}")
	bounds = get_boundaries(conn=connections, boundary_query=bound_query)

	logger.setLevel(logging.DEBUG)
	if logger.isEnabledFor(logging.DEBUG):
		logger.debug(f"Bounds: {bounds}, numPartitions: {numPartitions}")
		logger.debug(f"Redshift connections: {connections}")
		logger.setLevel(logging.INFO)

	# Reading from Redshift
	logger.info(f"Fetching Raw data from Redshift:\n {query}")
	df = (spark.read.format("jdbc")
		.option("dbtable", query)
		.option("driver", connections["driver"])
		.option("url", connections["url"])
		.option("user", connections["user"])
		.option("password", connections["password"])
		.option("partitionColumn", PARTITION_COLUMN)
		.option("lowerBound", bounds.min)
		.option("upperBound", bounds.max + 1)
		.option("numPartitions", numPartitions)
		.load())

	# Writing to Google Cloud Storage
	df.write.mode("overwrite").parquet(FULL_FILENAME)
	logger.info(f"Raw data saved successfully in Google Cloud Storage: {FULL_FILENAME}")


if __name__ == "__main__":
	# Initiating Spark Session
	spark = SparkSession \
		.builder \
		.getOrCreate()

	main()
	logger.info("Job ended Successfully.")
