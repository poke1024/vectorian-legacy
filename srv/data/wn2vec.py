import os
import pyarrow.parquet as pq

from .utils import prepare_apsynp, make_table, prepare_neighborhood, prepare_percentiles


def _prepare_wn2vec(path, parquet_path, config):
	import numpy

	embeddings = numpy.load(os.path.join(path, "embeddings_matrix.npy"))
	tokens = numpy.load(os.path.join(path, "word_list.npy"))

	if 'apsynp' in config.metrics:
		prepare_apsynp(embeddings, parquet_path)

	from sklearn.preprocessing import normalize

	embeddings = normalize(embeddings, axis=1, norm='l2')
	embeddings = numpy.clip(embeddings, 0, 1)

	if 'nicdm' in config.metrics:
		prepare_neighborhood(embeddings, parquet_path)

	import pyarrow.parquet as pq

	pq.write_table(
		make_table(tokens, embeddings),
		parquet_path + ".parquet",
		compression='snappy',
		version='2.0')


def load(vcore, config):
	print("loading wn2vec.", flush=True)

	base_path = os.path.dirname(os.path.realpath(__file__))
	data_path = os.path.join(base_path, "..", "..", "data")

	parquet_path = os.path.join(data_path, "cache", "wn2vec")

	if not os.path.exists(parquet_path + ".parquet"):
		_prepare_wn2vec(os.path.join(data_path, "wn2vec"), parquet_path, config)

	print("loading fasttext parquet table...")
	vec_table = pq.read_table(parquet_path + ".parquet")
	print("done.")

	embedding = vcore.FastEmbedding("wn2vec", vec_table)

	if 'percentiles' in config.metrics:
		prepare_percentiles(embedding, parquet_path)

	return embedding

