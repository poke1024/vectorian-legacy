from. utils import *
import numpy
import collections
import numbers


def evaluate(grid, measures, topics, matrix_path, on_done=None):
	from .evaluator import Evaluator
	import logging

	logging.getLogger('pykka').setLevel(logging.ERROR)

	evaluator = Evaluator.start(grid, measures, topics, matrix_path, on_done).proxy()

	evaluator.next_search()  # initial trigger

	return evaluator


def diff_1(a, b):
	candidate = None

	for i, (x, y) in enumerate(zip(a, b)):
		if x != y:
			if candidate is None:
				candidate = i
			else:
				return None

	return candidate


def merge_dimension(l, n):
	d = collections.defaultdict(list)

	for x in l:
		d[tuple(x[:n] + x[n + 1:])].append(x[n])

	return [(list(k[:n]) + [", ".join([str(x) for x in sorted(v)])] + list(k[n:])) for k, v in d.items()]


def merge_all_dimensions(l):
	space = [(i, len(set([x[i] for x in l]))) for i in range(len(l[0]))]
	space = sorted(space, key=lambda s: s[1], reverse=True)

	for i, _ in space:
		l = merge_dimension(l, i)

	return l


def _to_tuple(x):
	if type(x) is list:
		return tuple(x)
	else:
		return x


def _tuplify(x):
	for p in x:
		yield tuple(map(_to_tuple, p))


def pp_metric(m):
	assert len(m) == 1
	m = m[0]

	alpha = m[-1]
	if isinstance(alpha, numbers.Integral):
		return m[alpha]
	else:
		return "-".join(map(str, m))


_default_formatters = dict(metrics=pp_metric)


def _bulma_table_head_title(name):
	short_name = "".join([s[0] for s in name.split("_")]).upper()
	return '<th><abbr title="%s">%s</abbr></th>' % (name, short_name)


class Evaluation:
	def __init__(self, grid, measures, results):
		self._grid = grid
		self._measures = measures

		import numpy
		complete_results = []
		for r in results:
			if numpy.all(r > -1.0):
				complete_results.append(r)

		self._results = complete_results

		if self._results:
			self._mean = numpy.mean(numpy.array(self._results), axis=0)
		else:
			self._mean = results[0]

	@staticmethod
	def read(path=None):
		import os

		if path is None:
			path = os.path.join(os.path.dirname(
				os.path.realpath(__file__)),
				"..", "..", "data", "evaluation")

		if not os.path.exists(path):
			return None

		grid = Grid(os.path.join(path, "grid.yml"))
		measures = Measures(os.path.join(path, "measures.yml"))
		frames = []

		matrix_path = os.path.join(path, "evaluated")
		for matrix_name in os.listdir(matrix_path):
			p = os.path.join(matrix_path, matrix_name)
			if os.path.isfile(p):
				array = numpy.memmap(
					p,
					dtype='float32',
					mode='r',
					shape=(grid.size, measures.size))
				frames.append(array)

		return Evaluation(grid, measures, frames)

	@property
	def grid(self):
		return self._grid

	@property
	def measures(self):
		return self._measures

	@property
	def frames(self):
		import pandas

		return map(lambda r: pandas.DataFrame(
			r, columns=self.measures.names), self._results)

	@property
	def mean(self):
		return self._mean

	def _df_to_bulma_table(self, df, columns):
		html = list()

		html.append('<table class="table is-fullwidth is-hoverable">')
		html.append('<thead>')
		html.append('<tr>')

		for x in columns:
			html.append(_bulma_table_head_title(x))

		html.append('</tr>')
		html.append('</thead>')

		html.append('<tbody>')

		for i in range(len(df)):
			row = df.iloc[i]

			html.append('<tr>')

			for j in range(len(row)):
				html.append("<td>%s</td>" % str(row.iloc[j]))

			html.append('</tr>')

		html.append('</tbody>')

		html.append('</table>')

		return ''.join(html)

	def optimal(self, formatters=_default_formatters, as_html=True):
		gr = list(_tuplify(self.grid.cartesian()))
		results = []

		row_formatters = []
		for i, p in enumerate(self.grid.parameters()):
			if p in formatters:
				row_formatters.append(formatters[p])
			else:
				row_formatters.append(str)

		def format_row(row):
			return [f(x) for f, x in zip(row_formatters, row)]

		for measure_i, measure_name in enumerate(self.measures.names):
			values = self.mean[:, measure_i]
			best = numpy.argmax(values)

			best_value = values[best]

			best_tied = numpy.array(range(len(values)), dtype=numpy.int32)[
				values >= best_value]

			best = [([measure_name, best_value] + format_row(gr[i])) for i in best_tied]
			best = merge_all_dimensions(best)

			results.append(best)

		import itertools
		results = list(itertools.chain(*results))
		columns = ["measure", "score"] + list(self.grid.parameters())

		import pandas as pd
		df = pd.DataFrame(results, columns=columns)

		if as_html:
			return self._df_to_bulma_table(df, columns)
		else:
			return df





# from Jupyter:
#
# import sys
# sys.path.append("/path/to/vectorian/srv")
#
# from evaluation import Evaluation
# e = Evaluation.read("/path/to/vectorian/misc/eval-demo")
# e.optimal()