# run as: python nicdm.py > nicdm.log 2>&1 &

import scipy.stats
import multiprocessing
import numpy as np
import concurrent.futures
import os

from sklearn.neighbors import BallTree
from tqdm import tqdm
from numba import jit

import multiprocessing
from pathlib import Path


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

		print("building ball tree.", flush=True)
		self._tree = BallTree(embeddings, leaf_size=2)
		print("done.", flush=True)

	def compute(self, i0):
		distances, indices = self._tree.query(
			embeddings[i0:i0 + self._chunk_size, :],
			self._k + 1,
			sort_results=True,
			dualtree=True)

		r = np.empty(shape=(len(distances), self._k), dtype=np.float32)
		return slice(i0, i0 + self._chunk_size), to_means(r, i0, self._k, distances, indices)


def _table_from_numpy(array):
	import pyarrow as pa

	vecs = [pa.array(array[:, i]) for i in range(array.shape[1])]
	vecs_name = [('v%d' % i) for i in range(array.shape[1])]

	return pa.Table.from_arrays(vecs, vecs_name)


def prepare_neighborhood(embeddings, parquet_path):
	c = Compute(embeddings)
	n_tokens = len(embeddings)

	print("found %d tokens." % n_tokens, flush=True)

	pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)

	neighborhood_distance = np.empty(shape=(n_tokens, c._k), dtype=np.float32)
	print("table size: %d %d" % neighborhood_distance.shape)

	items = list(range(0, n_tokens, c._chunk_size))
	for i, (y, r) in tqdm(enumerate(pool.imap_unordered(c.compute, items)), total=len(items)):
		neighborhood_distance[y, :] = r

	import pyarrow.parquet as pq

	print("writing table to %s." % (str(parquet_path) + ".neighborhood.parquet"), flush=True)

	pq.write_table(
		_table_from_numpy(neighborhood_distance),
		str(parquet_path) + ".neighborhood.parquet",
		compression='snappy',
		version='2.0')

	print("done.", flush=True)


filename = "crawl-300d-2M-subword.vec"

script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
data_path = script_dir.parent.parent / "data"

tokens, embeddings = _load_embeddings(data_path / "fasttext" / filename)

#embeddings = np.random.rand(50000, 100)

parquet_path = data_path / "cache" / filename
prepare_neighborhood(embeddings, parquet_path)


