import pykka
import traceback
import sys
import os

from. utils import *

class Evaluator(pykka.ThreadingActor):
	def __init__(self, grid, measures, topics, matrix_path, on_done=None):
		super().__init__()
		self._proxy = None

		self._grid = grid
		self._measures = measures
		self._topics = topics
		assert len(topics) > 0
		self._matrix_path = matrix_path
		self._on_done = on_done

		self._current_topic_index = 0
		self._matrix = None
		self._parameters = None
		self._pending = 0

		from tqdm import tqdm
		self._tqdm = tqdm(
			desc='evaluation',
			total=grid.size * len(topics))
		self._tqdm.display = lambda **kwargs: None

		import humanize

		print("there are %d topics, grid size is %d." % (len(topics), grid.size))
		print("matrix size will be about %s." % humanize.naturalsize(
			len(topics) * grid.size * 4))

		import datetime

		print("with 1s per query, this will be done %s." % humanize.naturaltime(
			datetime.timedelta(seconds=-len(self._tqdm))))

		self._start_topic()

	def on_start(self):
		self._proxy = self.actor_ref.proxy()

	def _start_topic(self):
		self._parameters = enumerate(self._grid.cartesian())

		if self._matrix:
			self._matrix.close()

		filename = os.path.join(
			self._matrix_path,
			"topic_%d.dat" % (1 + self._current_topic_index))

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

				self._current_topic_index += 1

				if self._current_topic_index >= len(self._topics):
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
