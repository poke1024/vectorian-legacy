import cpp as vcore
import json
import time

import traceback
import sys
import os
import shutil
import argparse
import sys
from datetime import datetime

import logging

from pathlib import Path


if int(os.environ.get('VERBOSE_VECTORIAN', 0)) != 0:
	logging.basicConfig(level=logging.DEBUG)
else:
	logging.basicConfig(level=logging.INFO)

from abacus import Abacus
from config import Config

import data.fasttext
import data.wn2vec
import data.corpus

import evaluation


class Task:
	def __init__(self, doc, query):
		self._doc = doc
		self._query = query

	def __call__(self):
		x = self._doc.find(self._query)
		return x

	@property
	def n_tokens(self):
		return self._doc.n_tokens


class Find:
	def __init__(self, query):
		self._query = query

	@staticmethod
	def get_location_desc(metadata, location):
		if location[2] > 0:  # we have an act-scene-speakers structure.
			speaker = metadata["speakers"].get(str(location[2]), "")
			if location[0] >= 0:
				acts = ("I", "II", "III", "IV", "V", "VI", "VII")
				act = acts[location[0]]
				scene = location[1]
				return speaker, "%s.%d, line %d" % (act, scene, location[3])
			else:
				return speaker, "line %d" % location[3]
		elif location[1] > 0:  # book, chapter and paragraphs
			if location[0] < 0:  # do we have a book?
				return "", "Chapter %d, par. %d" % (location[1], location[3])
			else:
				return "", "Book %d, Chapter %d, par. %d" % (
					location[0], location[1], location[3])
		else:
			return "", "par. %d" % location[3]

	def __call__(self, doc):
		return Task(doc, self._query)


def _build_matches(result_set):
	matches = []
	for i, m in enumerate(result_set.best_n(100)):

		regions = []

		try:
			for r in m.regions:
				s = r.s.decode('utf-8', errors='ignore')
				if r.matched:
					t = r.t.decode('utf-8', errors='ignore')
					regions.append(dict(
						s=s,
						t=t,
						similarity=r.similarity,
						weight=r.weight,
						pos_s=r.pos_s.decode('utf-8', errors='ignore'),
						pos_t=r.pos_t.decode('utf-8', errors='ignore'),
						metric=r.metric.decode('utf-8', errors='ignore')))
				else:
					regions.append(dict(s=s, mismatch_penalty=r.mismatch_penalty))

			metadata = m.document.metadata
			speaker, loc_desc = Find.get_location_desc(metadata, m.location)

			matches.append(dict(
				debug=dict(document=m.document.id, sentence=m.sentence_id),
				score=m.score,
				algorithm=m.metric,
				location=dict(
					speaker=speaker,
					author=metadata["author"],
					work=metadata["title"],
					location=loc_desc
				),
				regions=regions,
				omitted=m.omitted))
		except UnicodeDecodeError as e:
			print(e)  # FIXME
			pass

	return matches


def _batanovic_weighting(strength):
	_stss_weights = dict([
		('CC', '0.7'), ('CD', '0.8'), ('DT', '0.7'), ('EX', '0.7'), ('FW', '0.7'), ('IN', '0.7'), ('JJ', '0.7'),
		('JJR', '0.7'), ('JJS', '0.8'), ('LS', '0.7'), ('MD', '1.2'), ('NN', '0.8'), ('NNS', '1.0'), ('NNP', '0.8'),
		('NNPS', '0.8'), ('PDT', '0.7'), ('POS', '0.7'), ('PRP', '0.7'), ('PRP$', '0.7'), ('RB', '1.3'), ('RBR', '1.2'),
		('RBS', '1.0'), ('RP', '1.2'), ('SYM', '0.7'), ('TO', '0.8'), ('UH', '0.7'), ('VB', '1.2'), ('VBD', '1.2'),
		('VBG', '1.1'), ('VBN', '0.8'), ('VBP', '1.2'), ('VBZ', '1.2'), ('WDT', '0.7'), ('WP', '0.7'), ('WP$', '0.7'),
		('WRB', '1.3')])

	pos_weights = dict()
	for p, w in _stss_weights.items():
		pos_weights[p] = (1 - strength) + (strength * float(w))

	return pos_weights


class Session(Abacus):
	def __init__(self, app, vocab, ws_send):
		super().__init__()
		self._app = app
		self._vocab = vocab
		self.ws_send = ws_send

		self._query = None

		self._set = None
		self._n_done = 0

		self._last_flush = time.time()
		self._report_progress = False

		self._config = Config()

	def _flush_set(self):
		if self._set is not None:
			data = dict(
				command='add-results',
				results=_build_matches(self._set))

			if self._report_progress:
				data["progress"] = self._n_done / self._app.n_tokens_in_corpus
			else:
				data["progress"] = 0

			self.ws_send(json.dumps(data))

			self._set = None
			self._last_flush = time.time()

	def _metrics(self, data):
		if data['enable_elmo'] and 'elmo' in self._config.embeddings:
			return ['elmo']
		elif 'wn2vec' in self._config.embeddings:
			return [('fasttext', 'wn2vec', float(data['mix_embedding']) / 100)]
		else:
			return ['fasttext']

	def _submit_search(self, data):
		if self._query is not None:
			logging.info("ignoring submit. search is still running.")
			return

		query_text = data['query']

		pos_mismatch = float(data['pos_mismatch']) / 100
		pos_weighting = float(data['pos_weighting']) / 100

		self._query = vcore.Query(
			self._vocab,
			query_text,
			self._app.query_tokens(query_text, data['ignore_determiners']).get_parquet_table(),
			metrics=self._metrics(data),
			pos_weights=_batanovic_weighting(pos_weighting),
			pos_mismatch_penalty=pos_mismatch,
			cost_combine_function=data['cost_combine_function'],
			mismatch_length_penalty=int(data['mismatch_length_penalty']),
			submatch_weight=data['submatch_weight'],
			idf_weight=float(data['idf_weight']) / 100,
			bidirectional=data['bidirectional'],
			similarity_threshold=(1 + float(data['similarity_threshold'])) / 100,
			similarity_falloff=float(data['similarity_falloff']),
			similarity_measure=data['similarity_measure'])

		self._set = None
		self._n_done = 0
		self._last_flush = time.time()
		self._report_progress = False

		self.submit(
			Find(self._query),
			self._app.documents)

	def on_ws_receive(self, data):
		logging.debug("Session.on_ws_receive %s" % str(data))

		if data['command'] == 'start-search':
			self.ws_send('search-started')
			try:
				self._submit_search(data)
			except Exception as e:
				self.ws_send('search-aborted')
				raise

		if data['command'] == 'abort-search':
			if self._query is not None:
				self._query.abort()
				if self.abort():
					self.ws_send('search-aborted')

	def on_aborted(self):
		logging.debug("Session.on_aborted")
		super().on_aborted()
		self._query = None
		self.ws_send('search-aborted')

	def on_finished(self):
		logging.debug("Session.on_finished")
		super().on_finished()
		self._flush_set()
		self._query = None
		self.ws_send('search-finished')

	def on_task_done(self, task, result):
		self._n_done += task.n_tokens

		if result is not None:
			if self._set is None:
				self._set = result
			else:
				self._set.extend(result)

		if time.time() - self._last_flush > 5:
			self._report_progress = True
			self._flush_set()


class BatchJob(Abacus):
	def __init__(self, app, query, on_finish):
		super().__init__()
		self._app = app
		self._query = query

		self._set = None
		self._n_done = 0

		self._last_report = time.time()
		self._on_finish = on_finish

		self.submit(
			Find(self._query),
			self._app.documents)

	def on_aborted(self):
		super().on_aborted()
		self._query = None
		print("BatchJob.aborted.", flush=True)
		self._on_finish(None)

	def on_finished(self):
		super().on_finished()
		self._query = None
		self._on_finish(self._set)

	def on_task_done(self, task, result):
		self._n_done += task.n_tokens

		if result is not None:
			if self._set is None:
				self._set = result
			else:
				self._set.extend(result)

		if time.time() - self._last_report > 1:
			self._last_report = time.time()


class Topic(evaluation.Topic):
	def __init__(self, app, query, truth):
		super().__init__(truth)
		self._app = app
		self._query_text = query

	def search(self, parameters, reply):
		options = dict(parameters.items())

		ignore_determiners = options['ignore_determiners']
		del options['ignore_determiners']

		options['metrics'] = [tuple(m) for m in options['metrics']]

		query = vcore.Query(
			self._app._vocab,
			self._query_text,
			self._app.query_tokens(
				self._query_text,
				ignore_determiners).get_parquet_table(),
			cost_combine_function='sum',
			**options)

		def query_done(result_set):
			# note: what we return here has to have to same format as self.truth, so
			# that things are comparable.

			if result_set:
				reply([(m.document.id, m.sentence_id) for m in result_set.best_n(100)])
			else:
				reply(None)

		BatchJob.start(self._app, query, query_done)


class App:
	def __init__(self):
		config = Config()

		print("loading embeddings.", flush=True)

		loaders = dict(
			fasttext=lambda: data.fasttext.load(vcore, config),
			wn2vec=lambda: data.wn2vec.load(vcore, config),
			elmo=lambda: vcore.ElmoEmbedding())

		vocab = vcore.Vocabulary()
		for embedding in config.embeddings:
			vocab.add_embedding(loaders[embedding]())

		self._vocab = vocab

		print("loading spacy.", flush=True)
		import spacy
		self._nlp = spacy.load("en_core_web_lg")
		data.corpus.configure_nlp(self._nlp)

		self._docs = data.corpus.documents(vocab, self._nlp)

		print("up.", flush=True)
		self._compute_stats()

		self._evaluator = None
		self._batch_processing()

	@property
	def documents(self):
		return self._docs

	@property
	def evaluator(self):
		return self._evaluator

	def start_session(self, ws_send):
		return Session.start(self, self._vocab, ws_send)

	def _compute_stats(self):
		n_sentences = 0
		n_tokens = 0
		for doc in self._docs:
			n_tokens += doc.n_tokens
			n_sentences += doc.n_sentences
		self.n_tokens_in_corpus = n_tokens
		print("corpus contains %s tokens in %s sentences." % (
			f'{n_tokens:,}', f'{n_sentences:,}'), flush=True)

	def query_tokens(self, query_text, ignore_det=False):
		tokens = self._nlp(query_text)

		if ignore_det:
			def token_filter(t):
				return t.pos_ != "DET"
		else:
			def token_filter(_):
				return True

		from data.corpus import TokenTable
		token_table = TokenTable()
		token_table.append(tokens, token_filter)

		return token_table

	def _batch_processing(self):
		# perform special batch commands here.

		parser = argparse.ArgumentParser(
			description='Vectorian corpus search.')
		parser.add_argument(
			'--dump', nargs='?', type=str, const="__default__", help='path to export corpus to')
		parser.add_argument(
			'--eval', nargs='?', type=str, const="__default__", help='evaluate the given yml file')
		args = parser.parse_args()

		data_path = Path(os.path.join(os.path.dirname(
			os.path.realpath(__file__)),
			"..", "data"))

		if args.dump:
			dump_path = args.dump
			if dump_path == "__default__":
				dump_path = data_path / "signatures"

			os.makedirs(dump_path, exist_ok=True)
			data.corpus.dump_corpus(self._docs, dump_path)

			print("export done.")
			sys.exit(0)

		batches_path = data_path / "batches"

		incoming_path = batches_path / "incoming"
		running_path = batches_path / "running"
		done_path = batches_path / "done"

		os.makedirs(incoming_path, exist_ok=True)
		os.makedirs(running_path, exist_ok=True)
		os.makedirs(done_path, exist_ok=True)

		if args.eval == "__default__":
			args.eval = None
			assert incoming_path.is_dir()
			for f in incoming_path.iterdir():
				p = incoming_path / f
				if p.is_dir():
					shutil.move(str(p), str(running_path))
					args.eval = running_path / f
					break

		if args.eval:
			basepath = Path(args.eval)

			topics = self._load_topics(basepath / "topics.yml")
			grid = evaluation.Grid(basepath / "grid.yml")
			measures = evaluation.Measures(basepath / "measures.yml")

			matrix_path = basepath / "evaluated"
			os.makedirs(matrix_path, exist_ok=True)

			def on_evaluation_finished():
				print("evaluation is finished.", flush=True)
				try:
					now = datetime.today().isoformat().replace(":", "-")
					shutil.move(str(basepath), str(done_path / (basepath.name + "-" + now)))
				except:
					traceback.print_exc()

			self._evaluator = evaluation.evaluate(
				grid, measures, topics, matrix_path, on_evaluation_finished)

	def _load_topics(self, filename):
		import yaml

		with open(filename, 'r') as f:
			topic_defs = yaml.safe_load(f)

		resolver = data.corpus.signature_resolver(
			self.documents, parser=topic_defs.get('parser'))

		print("loading topics.")
		topics = list()
		for topic_def in topic_defs['topics']:
			unknown_documents = set()
			unknown_sentences = list()

			def on_error(type, signature, document=None):
				if type == "document":
					unknown_documents.add(signature)
				elif type == "sentence":
					unknown_sentences.append((document, signature))

			truth = resolver(topic_def['truth'], on_error=on_error)
			if truth:
				n = len(topic_def['truth'])
				if len(truth) < n:
					print("[FAIL] %s" % topic_def['query'])
					print("  skipped %d ground truth references." % (n - len(truth)))
				else:
					print("[OK  ] %s" % topic_def['query'])
				topics.append(Topic(self, topic_def['query'], truth))
			else:
				print("[SKIP] %s" % topic_def['query'])

			if unknown_sentences:
				print("")
				print("  the following sentence signatures could not be resolved:")
				for doc, sig in unknown_sentences:
					print("  - %s in '%s' by %s" % (
						sig, doc.metadata['title'],  doc.metadata['author']))

			if unknown_documents:
				print("")
				print("  the following document signatures could not be resolved:")
				for sig in unknown_documents:
					print("  - %s" % sig)

			if unknown_sentences or unknown_documents:
				print("")

		return topics
