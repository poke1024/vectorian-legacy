import numpy
import yaml

from .measures import Measurable


class Grid:
	def __init__(self, grid):
		if type(grid) is not dict:
			raise RuntimeError("grid must be a dict")

		# guarantee defined order independent of dict.
		pairs = sorted(list(grid.items()), key=lambda p: p[0])
		self._parameters = tuple(map(lambda p: p[0], pairs))
		self._values = tuple(map(lambda p: p[1], pairs))

		size = 1
		for k, v in grid.items():
			size *= len(v)
		self._size = size

	@property
	def size(self):
		return self._size

	def parameters(self):
		return self._parameters

	def cartesian(self):
		import itertools
		return itertools.product(*self._values)


class Measures:
	def __init__(self, filename):
		with open(filename, 'r') as f:
			self._measures = yaml.safe_load(f)

	@property
	def size(self):
		return len(self._measures)

	@property
	def names(self):
		short_names = dict(
			average_precision='ap',
			discounted_cumulative_gain='dcg')

		names = []
		for m in self._measures:
			name = m['type']

			if name in short_names:
				name = short_names[name]

			for k, v in m.items():
				if k != 'type':
					if len(m) == 2:
						name += "@%s" % v
					else:
						name += ", %s=%s" % (k, v)
			names.append(name)
		return names

	def evaluate(self, search_result, truth):
		measured = numpy.zeros(
			(len(self._measures), ), dtype=numpy.float32)

		if search_result is not None:
			data = Measurable(search_result, truth)

			for i, m in enumerate(self._measures):
				measured[i] = data.compute(m)

		return measured


class Topic:
	def __init__(self, truth):
		self._truth = truth

	@property
	def truth(self):
		return self._truth

	def search(self, parameters, reply):
		raise NotImplementedError()


class Matrix:
	def __init__(self, grid, measures, filename):
		self._grid = grid
		self._measures = measures
		self._path = filename

		self._matrix = numpy.memmap(
			filename,
			dtype='float32',
			mode='w+',
			shape=(grid.size, measures.size))

		print("initializing on-disk matrix.", flush=True)
		self._matrix.fill(-1)
		self._matrix.flush()
		print("done", flush=True)

	def write(self, grid_index, search_result, search_truth):
		measured = self._measures.evaluate(
			search_result, search_truth)
		self._matrix[grid_index, :] = measured
		self._matrix.flush()

	def close(self):
		self.write_as_json()

		# this is a hack, as there's no official to do this
		# as of 2019/10/06.
		self._matrix._mmap.close()

	def write_as_json(self):
		cartesian = tuple(self._grid.cartesian())
		measures = tuple(self._measures.names)
		assert len(cartesian) == self._matrix.shape[0]
		assert len(measures) == self._matrix.shape[1]

		with open(self._path + ".csv", "w") as f:
			for m in measures:
				f.write(";%s" % m)
			f.write("\n")

			for grid_index, grid_params in enumerate(cartesian):
				f.write("%s;" % str(grid_params))
				row = []
				for measure_index in range(len(measures)):
					row.append("%f" % self._matrix[grid_index, measure_index])
				f.write(";".join(row) + "\n")
