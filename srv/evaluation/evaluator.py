import pykka
import traceback
import sys
import os
import humanize
import datetime
import asyncio
import numpy as np
import concurrent

from tqdm import tqdm
from pathlib import Path

from. utils import *


class Evaluator(pykka.ThreadingActor):
	def __init__(self, grid, measures, topics, matrix_path, on_done=None):
		super().__init__()
		self._proxy = None

		self._grid = grid
		self._measures = measures
		self._topics = topics
		print(f"got {len(topics)} valid topics.")
		assert len(topics) > 0

		self._matrix_path = Path(matrix_path)
		self._on_done = on_done

		self._current_topic_index = 0
		self._matrix = None
		self._parameters = None
		self._pending = 0

		self._tqdm = tqdm(
			desc='evaluation',
			total=grid.size * len(topics))
		self._tqdm.display = lambda **kwargs: None

		print("there are %d topics, grid size is %d." % (len(topics), grid.size))
		print("matrix size will be about %s." % humanize.naturalsize(
			len(topics) * grid.size * 4))

		print("with 1s per query, this will be done %s." % humanize.naturaltime(
			datetime.timedelta(seconds=-len(self._tqdm))))

		self._start_topic()

	def on_start(self):
		self._proxy = self.actor_ref.proxy()

	def _start_topic(self):
		self._parameters = enumerate(self._grid.cartesian())

		if self._matrix:
			self._matrix.close()

		filename = self._matrix_path / (
			"topic.%d" % (1 + self._current_topic_index))

		self._matrix = Matrix(
			self._grid, self._measures, filename)

	def tqdm(self):
		return self._tqdm

	def next_search(self):
		while self._pending < 2:
			try:
				grid_index, grid_params = next(self._parameters)

				def make_reply():
					search_id = (self._current_topic_index, grid_index)

					def reply(search_result):
						self._proxy.search_done(search_id, search_result)

					return reply

				self._pending += 1

				self._topics[self._current_topic_index].search(
					dict(zip(self._grid.parameters(), grid_params)), make_reply())

			except StopIteration:
				if self._pending > 0:
					# we must not switch topics or to a new self._matrix
					# while we still have a pending search that needs to
					# write to the current topic's matrix.
					break

				print("done with topic %d (total %d)." % (
					1 + self._current_topic_index, len(self._topics)))
				self._current_topic_index += 1

				if self._current_topic_index >= len(self._topics):
					print("writing matrix data...", flush=True)
					self._matrix.close()
					print("grid search is done.", flush=True)

					if self._on_done:
						self._on_done()
					return  # we're done

				self._start_topic()

			except:
				traceback.print_exc()
				sys.stdout.flush()

	def search_done(self, search_id, search_result):
		self._pending -= 1
		self._tqdm.update(1)

		topic_index, grid_index = search_id

		self._matrix.write(
			grid_index,
			search_result,
			self._topics[topic_index].truth)

		self.next_search()


class BlockingEvaluator(pykka.ThreadingActor):
	def __init__(self, measures, topics):
		super().__init__()
		self._proxy = None

		self._measures = measures
		self._topics = topics
		print(f"got {len(topics)} valid topics.")
		assert len(topics) > 0

		self._tqdm = tqdm(
			desc='evaluation',
			total=1)
		self._tqdm.display = lambda **kwargs: None

		self._queue = []
		self._results = []
		self._future = None

	def on_start(self):
		self._proxy = self.actor_ref.proxy()

	def evaluate(self, **params):
		print("evaluating parameters:", str(params))

		self._queue = []
		self._results = []

		for i in range(len(self._topics)):
			self._queue.append((i, params))

		self._future = concurrent.futures.Future()
		self.next_search()

		return self._future

	def tqdm(self):
		return self._tqdm

	def next_search(self):
		if not self._queue:
			return

		topic_index, params = self._queue.pop()

		def make_reply():
			def reply(search_result):
				self._proxy.search_done(topic_index, search_result)

			return reply

		self._topics[topic_index].search(params, make_reply())

	def search_done(self, topic_index, search_result):
		truth = self._topics[topic_index].truth

		measured = self._measures.evaluate(
			search_result, truth)
		self._results.append(measured)

		if not self._queue:
			self._future.set_result(
				np.average(self._results, axis=0))
		else:
			self.next_search()
