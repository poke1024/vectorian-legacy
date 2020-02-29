import scipy.stats
import multiprocessing
from joblib import Parallel, delayed


import scipy.stats
import multiprocessing
import numpy as np
import concurrent.futures

from sklearn.neighbors import BallTree
from tqdm import tqdm
from numba import jit

import multiprocessing
import os


def _compute_inverse_ranking(vec):
	return (len(vec) + 1) - scipy.stats.rankdata(vec, method='ordinal')


def make_table(tokens, embeddings):
	import pyarrow as pa

	vecs = [pa.array(embeddings[:, i]) for i in range(embeddings.shape[1])]
	vecs_name = [('v%d' % i) for i in range(embeddings.shape[1])]

	return pa.Table.from_arrays(
		[pa.array(tokens, type=pa.string())] + vecs,
		['token'] + vecs_name)


def _table_from_numpy(array):
	import pyarrow as pa

	vecs = [pa.array(array[:, i]) for i in range(array.shape[1])]
	vecs_name = [('v%d' % i) for i in range(array.shape[1])]

	return pa.Table.from_arrays(vecs, vecs_name)


def prepare_apsynp(embeddings, parquet_path):
	from tqdm import tqdm
	import numpy

	if os.path.exists(parquet_path + ".apsynp.parquet"):
		return

	apsynp = numpy.empty(
		shape=embeddings.shape, dtype=numpy.uint16)

	print("computing apsynp.")

	for i in tqdm(range(len(embeddings))):
		apsynp[i, :] = _compute_inverse_ranking(embeddings[i, :])

	import pyarrow.parquet as pq

	pq.write_table(
		_table_from_numpy(apsynp),
		parquet_path + ".apsynp.parquet",
		compression='snappy',
		version='2.0')


@jit(nopython=True)
def to_means(r, i0, k, distances, indices):
	for ii in range(len(distances)):
		i = i0 + ii  # queried token

		selected = np.extract(
			indices[ii] != i,
			distances[ii])

		x = 0.0
		for j in range(k):
			x += selected[j]
			r[ii, j] = x / max(j, 1)

	return r


class Compute:
	def __init__(self, embeddings):
		self._chunk_size = 500
		self._k = 100  # (maximum) number of nearest neighbors to average over

		print("building ball tree.")
		self._tree = BallTree(embeddings, leaf_size=2)
		print("done.")

		self._embeddings = embeddings

	def compute(self, i0):
		distances, indices = self._tree.query(
			self._embeddings[i0:i0 + self._chunk_size, :],
			self._k + 1,
			sort_results=True,
			dualtree=True)

		r = np.empty(shape=(len(distances), self._k), dtype=np.float32)
		return slice(i0, i0 + self._chunk_size), to_means(r, i0, self._k, distances, indices)


def prepare_neighborhood(embeddings, parquet_path):

	if os.path.exists(parquet_path + ".neighborhood.parquet"):
		return

	c = Compute(embeddings)
	n_tokens = len(embeddings)

	print("computing.")

	pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)

	neighborhood_distance = np.empty(shape=(n_tokens, c._k), dtype=np.float32)
	items = list(range(0, n_tokens, c._chunk_size))
	for i, (y, r) in tqdm(enumerate(pool.imap_unordered(c.compute, items)), total=len(items)):
		neighborhood_distance[y, :] = r

	import pyarrow.parquet as pq

	print("writing table.")

	pq.write_table(
		_table_from_numpy(neighborhood_distance),
		parquet_path + ".neighborhood.parquet",
		compression='snappy',
		version='2.0')

	print("done.")


def prepare_percentiles(embedding, parquet_path):
	import os

	for measure in embedding.measures:
		if not measure.startswith('ranked-'):
			continue

		if measure == "ranked-maximum":
			# ignore. we max over ranked measures and do not rank over maxed
			# measures.
			continue

		short_name = '-'.join(measure.split('-')[1:])

		percentiles_path = parquet_path + ".percentiles." + measure + ".parquet"

		if os.path.exists(percentiles_path):
			continue

		print("computing percentiles for %s." % short_name, flush=True)

		import numpy
		from tqdm import tqdm

		n_tokens = embedding.n_tokens
		k = 1000
		numpy.random.seed(47451)

		parts = []
		for _ in tqdm(range(1000)):
			s = numpy.random.randint(n_tokens, size=k, dtype=numpy.int32)
			t = numpy.random.randint(n_tokens, size=k, dtype=numpy.int32)
			m = embedding.similarity_matrix(short_name, s, t)
			parts.append(m)

		similarities = numpy.concatenate(parts, axis=None)
		print("sorting.")
		numpy.ndarray.sort(similarities)

		n_steps = 1000
		percentiles = numpy.empty(shape=(n_steps + 1, ), dtype=numpy.float32)
		for i in range(n_steps + 1):
			j = (i * (len(similarities) - 1)) // n_steps
			percentiles[i] = similarities[j]

		import pyarrow.parquet as pq
		import pyarrow as pa

		table = pa.Table.from_arrays(
			[pa.array(percentiles)],
			[measure])

		pq.write_table(
			table,
			percentiles_path,
			compression='snappy',
			version='2.0')

	embedding.load_percentiles(parquet_path)
