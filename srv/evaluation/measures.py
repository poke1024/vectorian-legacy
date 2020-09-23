import numpy
import math

# averaging AveragePrecision over many queries will give
# us mean average precision (MAP).


class Measurable:
	def __init__(self, recommended, relevant):
		self._recommended = recommended
		self._relevant = set(relevant)

		if len(self._relevant) < 1:
			raise ValueError("need at least 1 relevant item")

		self._rel_data = numpy.zeros(
			(max(len(self._recommended), len(relevant)), ), dtype=numpy.int8)
		for k, rec in enumerate(self._recommended):
			if rec in self._relevant:
				self._rel_data[k] = 1

		self._cum_rel_data = numpy.cumsum(
			self._rel_data, dtype=numpy.int32)

	def _rel(self, k):
		return self._rel_data[k - 1]

	def precision_at_k(self,  k):
		return self._cum_rel_data[k - 1] / k

	def roc(self, cutoff):
		# roc: TP/(TP + FN) and FP/(FP + TN)
		tp = self._cum_rel_data[cutoff - 1]
		fp = cutoff - tp
		#fp = 0

		n_total_rel = len(self._relevant)  # tp + fn
		return numpy.array([fp / (fp + tn), tp / n_total_rel])

	def average_precision(self, n):  # average precision
		if len(self._relevant) < 1:
			# cannot compute average precision without any relevant items
			return "?"

		assert n <= len(self._recommended)
		m = float(len(self._relevant))
		return sum(self.precision_at_k(k) * self._rel(k) for k in range(1, n + 1)) / m

	def discounted_cumulative_gain(self, n):
		if len(self._recommended) == 0:
			return 0
		eff_n = min(n, len(self._recommended))
		return sum(self._rel(i) / math.log2(i + 1) for i in range(1, eff_n + 1))

	def normalized_discounted_cumulative_gain(self, n):
		x = self.discounted_cumulative_gain(n)
		k = min(len(self._relevant), n)
		best = sum(1 / math.log2(i + 1) for i in range(1, k + 1))
		return x / best

	def r_precision(self):
		R = len(self._relevant)
		return self._cum_rel_data[R - 1] / float(R)

	def compute(self, args):
		method = getattr(self, args['type'])

		kwargs = dict()
		for k, v in args.items():
			if k != 'type':
				kwargs[k] = v

		return method(**kwargs)
