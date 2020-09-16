# a data tool to write fasttext into a parquet file.

import os
from .utils import prepare_apsynp, make_table, prepare_neighborhood, prepare_percentiles
import pyarrow.parquet as pq


def _load_embeddings(csv_path):
	from tqdm import tqdm
	import numpy
	from sklearn.preprocessing import normalize

	tokens = []
	with open(csv_path, "r") as f:
		n_rows, n_cols = map(int, f.readline().strip().split())

		embeddings = numpy.empty(
			shape=(n_rows, n_cols), dtype=numpy.float64)

		for _ in tqdm(range(n_rows)):
			values = f.readline().strip().split()

			t = values[0]
			if t.isalpha() and t.lower() == t:

				embeddings[len(tokens), :] = values[1:]

				tokens.append(t)

	embeddings = embeddings[:len(tokens), :]

	return tokens, embeddings


def _prepare_fasttext(csv_path, parquet_path, config):
	from tqdm import tqdm
	import numpy
	from sklearn.preprocessing import normalize

	tokens, embeddings = _load_embeddings(csv_path)

	if 'apsynp' in config.metrics:
		prepare_apsynp(embeddings, parquet_path)

	embeddings = normalize(embeddings, axis=1, norm='l2')
	embeddings = embeddings.astype(numpy.float32)

	import pyarrow.parquet as pq

	pq.write_table(
		make_table(tokens, embeddings),
		parquet_path + ".parquet",
		compression='snappy',
		version='2.0')

	if 'nicdm' in config.metrics:
		prepare_neighborhood(embeddings, parquet_path)


def load(vcore, config):
	print("loading fasttext.", flush=True)

	#filenames = [
	#	"crawl-300d-2M-subword.vec",
	#	"wiki-news-300d-1M-subword.vec"]

	base_path = os.path.dirname(os.path.realpath(__file__))
	data_path = os.path.realpath(os.path.join(base_path, "..", "..", "data"))

	os.makedirs(os.path.join(data_path, "cache"), 0o777, True)

	filename = config.fasttext

	parquet_path = os.path.join(data_path, "cache", filename)

	if not os.path.exists(parquet_path + ".parquet"):
		csv_path = os.path.join(data_path, "fasttext", filename)
		print("creating %s from %s" % (parquet_path, csv_path), flush=True)
		_prepare_fasttext(csv_path, parquet_path, config)

	#t0 = time.time()
	print("loading fasttext parquet table...")
	vec_table = pq.read_table(parquet_path + ".parquet")
	print("done.")

	embedding = vcore.FastEmbedding("fasttext", vec_table)

	if 'apsynp' in config.metrics:
		apsynp_path = parquet_path + ".apsynp.parquet"
		embedding.add_apsynp(pq.read_table(apsynp_path), 0.1)

	if 'nicdm' in config.metrics:
		nicdm_path = parquet_path + ".neighborhood.parquet"
		embedding.add_nicdm(pq.read_table(nicdm_path))

	if 'percentiles' in config.metrics:
		prepare_percentiles(embedding, parquet_path)

	return embedding
