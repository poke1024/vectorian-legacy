import pykka
import collections
import time
import sys
import platform
import logging
import multiprocessing
import psutil
import threading
import humanize

cpu_load_limit = 0.8  # cpu load we aim for.


def print_debug_stats():
	lines = []

	lines.append("memory:")
	lines.append("  physical: %s" % humanize.naturalsize(psutil.virtual_memory().used))
	lines.append("  swap: %s" % humanize.naturalsize(psutil.swap_memory().used))

	lines.append(f"running {threading.active_count()} threads:")
	for thread in threading.enumerate():
		lines.append("  " + thread.name)

	logging.info("\n".join(lines))


class Batch(pykka.ThreadingActor):
	def __init__(self, abacus, compute, tasks, verbose=False):
		super().__init__()
		self._abacus = abacus
		self._compute = compute
		self._tasks = iter(tasks)
		self._aborted = False
		self._start_time = time.time()
		self._verbose = verbose

	def on_stop(self):
		if False:
			print("batch is closed.")

	def abort(self):
		self._aborted = True

	def fetch_next_task(self):
		logging.debug("batch is fetching task.")
		if self._aborted:
			return None
		try:
			item = next(self._tasks)
			return self._compute(item)  # returns a lambda
		except StopIteration:
			return None

	def on_task_done(self, task, result):
		self._abacus.on_task_done(task, result)

	def on_finished(self):
		if self._verbose:
			logging.info("batch finished after %.1f s on %d cpus (%s)." % (
				time.time() - self._start_time,
				multiprocessing.cpu_count(),
				"aborted" if self._aborted else "success"))
			print_debug_stats()
		else:
			print(".", flush=True, end="")

		if self._aborted:
			self._abacus.on_aborted()
		else:
			self._abacus.on_finished()


class Abacus(pykka.ThreadingActor):
	dispatch = None

	@staticmethod
	def delete_dispatcher():
		logging.debug("deleting dispatcher.")
		Abacus.dispatch = None

	def __init__(self):
		super().__init__()
		self.batch = None

	def on_stop(self):
		if self.batch:
			self.batch.abort()
		# print("abacus is closed.")

	def abort(self):
		if self.batch:
			self.batch.abort()
			return False
		else:
			return True

	def submit(self, compute, tasks, verbose=False):
		if self.batch:
			raise RuntimeError("cannot submit multiple batches")

		if not tasks:
			self.on_finished()
			return

		if Abacus.dispatch is None:
			logging.debug("creating dispatcher.")
			Abacus.dispatch = Dispatch.start().proxy()

		self.batch = Batch.start(
			self.actor_ref.proxy(), compute, tasks, verbose).proxy()
		Abacus.dispatch.add_batch(self.batch)

	def on_task_done(self, task, result):
		pass

	def on_finished(self):
		self.batch = None

	def on_aborted(self):
		self.batch = None


class Worker(pykka.ThreadingActor):
	def __init__(self, worker_id, master_proxy):
		super().__init__()
		self._master = master_proxy
		self._worker_id = worker_id

	def work_on_task(self, batch, task):
		result = task()
		self._master.on_task_result(self._worker_id, batch, task, result)

	def shutdown(self):
		self.stop()

	def on_failure(self, exception_type, exception_value, traceback):
		print("on_failure", exception_type, exception_value, traceback)
		sys.stdout.flush()

		self._master.on_worker_failure(self._worker_id)


class Dispatch(pykka.ThreadingActor):
	def __init__(self, n_workers=None):
		super().__init__()

		if n_workers is None:
			n_cpus = multiprocessing.cpu_count()

			import math
			n_workers = math.floor(n_cpus * cpu_load_limit)

		self._n_workers = n_workers
		self._workers = dict()
		self._proxy = None
		self._batches = collections.deque()
		self._n_tasks = collections.defaultdict(int)
		self._shutdown = False

	def on_start(self):
		self._proxy = self.actor_ref.proxy()

	def add_batch(self, batch):
		self._batches.append(batch)
		self._start_workers()

	def _send_next_task(self, worker_id):
		batch, task = self._fetch_next_task()
		if task is None:
			self._remove_worker(worker_id)
		else:
			self._n_tasks[id(batch)] += 1
			self._workers[worker_id].work_on_task(batch, task)

	def _fetch_next_task(self):
		while self._batches:
			batch = self._batches[0]
			task = batch.fetch_next_task().get()
			if task is not None:
				return batch, task
			self._batches.popleft()

		return None, None

	def _start_workers(self):
		logging.debug("starting workers.")
		for i in range(self._n_workers):
			if i not in self._workers:
				w = Worker.start(i, self._proxy)
				self._workers[i] = w.proxy()
				self._send_next_task(i)

	def _remove_worker(self, worker_id):
		if worker_id in self._workers:
			self._workers[worker_id].shutdown()
			del self._workers[worker_id]

		if self._shutdown and not self._workers:
			self.stop()
			Abacus.delete_dispatcher()

	def shutdown(self):
		if not self._workers:
			self.stop()
			Abacus.delete_dispatcher()
		else:
			self._shutdown = True

	def on_task_result(self, worker_id, batch, task, result):
		batch.on_task_done(task, result)
		self._send_next_task(worker_id)

		self._n_tasks[id(batch)] -= 1
		if self._n_tasks[id(batch)] == 0:
			del self._n_tasks[id(batch)]
			batch.on_finished()
			batch.stop()
			#print_debug_stats()

	def on_worker_failure(self, worker_id):
		# log?
		self._send_next_task(worker_id)
