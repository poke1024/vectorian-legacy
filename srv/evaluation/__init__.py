import numpy as np
import collections
import numbers
import pykka.messages
import yaml
import time
import logging
import threading

from pathlib import Path
from. utils import *

from .evaluator import Evaluator, BlockingEvaluator
from abacus import print_debug_stats


class Computer:
	def __init__(self, measures, topics):
		self._evaluator = BlockingEvaluator.start(measures, topics)

	def __call__(self, args):
		args = args.copy()

		args["metrics"] = [["fasttext", "wn2vec", args["mix_fasttext_wn2vec"]]]
		del args["mix_fasttext_wn2vec"]

		result = self._evaluator.ask(
			pykka.messages.ProxyCall(
				attr_path=['evaluate'],
				args=[],
				kwargs=args
			)
		)

		print("waiting for result.", flush=True, end="")
		t0 = time.time()
		scores = result.result()
		print(" done (%.1fs)." % (time.time() - t0), flush=True)

		print_debug_stats()

		score = np.average(scores)

		return score


class Objective:
	def __init__(self, measures, topics, params, outfile):
		self._computer = Computer(measures, topics)
		self._params = params
		self._outfile = outfile

	def __call__(self, trial):
		args = dict()
		for param in self._params:
			name = param["name"]
			t = param["type"]
			if t == "boolean":
				args[name] = trial.suggest_categorical(
					name, ['false', 'true']) == 'true'
			elif t == "categorical":
				args[name] = trial.suggest_categorical(
					name, param["choices"])
			elif t == "uniform":
				args[name] = trial.suggest_uniform(
					name, param["low"], param["high"])
			elif t == "int":
				args[name] = trial.suggest_int(
					name, param["low"], param["high"])
			else:
				raise ValueError("illegal parameter type " + t)

		score = self._computer(args)

		self._outfile.write("%d;trial;%f;%s\n" % (
			trial.number, score, str(args)))
		self._outfile.flush()

		return score


def _optuna_optimize(config, measures, topics, basepath):
	import optuna

	study_args = config["study"]

	if "sampler" in config:
		sampler_data = config["sampler"]
		sampler_type = sampler_data["type"]  # e.g. TPESampler, RandomSampler
		study_args["sampler"] = getattr(optuna.samplers, sampler_type)(
			**sampler_data.get("args", dict()))

	study = optuna.create_study(direction="maximize", **study_args)

	with open(basepath / "evaluation_core.csv", "w") as f:
		objective = Objective(measures, topics, config["parameters"], f)

		study.optimize(
			objective, n_trials=int(config["n_trials"]))

		f.write("%d;best;%f;%s\n" % (
			study.best_trial.number, study.best_value, str(study.best_params)))
		f.flush()

	df = study.trials_dataframe()
	df.to_csv(basepath / "evaluation_full.csv")

	print("optuna done.", flush=True)


def gen_ablations(param):
	name = param["name"]
	t = param["type"]
	if t == "boolean":
		yield name, False
		yield name, True
	elif t == "categorical":
		for c in param["choices"]:
			yield name, c
	elif t == "uniform":
		for x in np.linspace(param["low"], param["high"], 101):
			yield name, x
	elif t == "int":
		for x in range(param["low"], param["high"] + 1):
			yield name, x
	else:
		raise ValueError("illegal parameter type " + t)


def _ablation_study(config, measures, topics, basepath):
	computer = Computer(measures, topics)

	with open(basepath / "ablation.csv", "w") as f:

		baseline_args = config["baseline"]
		baseline_score = computer(baseline_args)
		f.write(f"baseline;{baseline_score}\n")
		f.flush()

		for param in config["parameters"]:  # ablation on param
			for name, value in gen_ablations(param):
				if baseline_args[name] == value:
					f.write(f"ablation;{name};{value};{baseline_score}\n")
				else:
					args = baseline_args.copy()
					args[name] = value
					score = computer(args)
					f.write(f"ablation;{name};{value};{score}\n")
				f.flush()

	print("ablation done.")


def evaluate(config, measures, topics, basepath, on_done=None):
	# set this to logging.DEBUG to debug strange hangs/aborts.
	logging.getLogger('pykka').setLevel(logging.ERROR)

	strategy = config["strategy"]
	if strategy == "optuna":
		t = threading.Thread(
			target=_optuna_optimize, args=(config, measures, topics, basepath))
		t.start()
		return None

	elif strategy == "ablation":
		t = threading.Thread(
			target=_ablation_study, args=(config, measures, topics, basepath))
		t.start()
		return None

	elif strategy == "grid":
		matrix_path = basepath / "evaluated"
		matrix_path.mkdir(exist_ok=True)

		new_evaluator = Evaluator.start(
			Grid(config["grid"]), measures, topics, matrix_path, on_done).proxy()

		new_evaluator.next_search()  # initial trigger

		return new_evaluator

	else:
		raise ValueError(f"illegal strategy {strategy}")


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
		frames = []
		matrix_path = os.path.join(path, "evaluated")
		for matrix_name in os.listdir(matrix_path):
			p = os.path.join(matrix_path, matrix_name)
			if os.path.isfile(p):
				array = np.memmap(
					p,
					dtype='float32',
					mode='r',
					shape=(grid.size, measures.size))
				frames.append(array)


		self._grid = grid
		self._measures = measures

		import numpy
		complete_results = []
		for r in results:
			if np.all(r > -1.0):
				complete_results.append(r)

		self._results = complete_results

		if self._results:
			self._mean = np.mean(np.array(self._results), axis=0)
		else:
			self._mean = results[0]

	@staticmethod
	def read(path=None):
		import os

		if path is None:
			path = os.path.join(os.path.dirname(
				os.path.realpath(__file__)),
				"..", "..", "data", "evaluation")

		path = Path(path)

		if not path.exists():
			return None

		with open(path / "config.yml", 'r') as f:
			config = yaml.safe_load(f)

		measures = Measures(path / "measures.yml")

		return Evaluation(config, measures)

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
			best = np.argmax(values)

			best_value = values[best]

			best_tied = np.array(range(len(values)), dtype=np.int32)[
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
