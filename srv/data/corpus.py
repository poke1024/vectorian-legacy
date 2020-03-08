import spacy
import pyarrow as pa
import pyarrow.parquet as pq
import pandas
import os
import re
import json
from functools import partial
import cpp as vcore


def _current_parser():
	return 'spacy-%s' % spacy.__version__


def ignore_token(t):
	return t.pos_ == 'PUNCT'


class TokenTable:
	def __init__(self):
		self._utf8_idx = 0

		self._token_idx = []
		self._token_len = []
		self._token_pos = []  # pos_ from spacy's Token
		self._token_tag = []  # tag_ from spacy's Token
		self._token_prob = []  # prob from spacy, see https://github.com/explosion/spaCy/issues/3874

	def __len__(self):
		return len(self._token_idx)

	def append(self, doc, filter=None):
		text = doc.text
		last_idx = 0

		for tokens in doc.sents:
			for token in tokens:
				if filter and not filter(token):
					continue

				self._utf8_idx += len(text[last_idx:token.idx].encode('utf8'))
				last_idx = token.idx

				if ignore_token(token):
					continue

				self._token_idx.append(self._utf8_idx)
				self._token_len.append(len(token.text.encode('utf8')))
				self._token_pos.append(token.pos_)
				self._token_tag.append(token.tag_)
				self._token_prob.append(token.prob)

		self._utf8_idx += len(text[last_idx:].encode('utf8'))

	def get_dataframe(self):
		return pandas.DataFrame({
			'idx': pandas.Series(self._token_idx, dtype='uint32'),
			'len': pandas.Series(self._token_len, dtype='uint8'),
			'pos': pandas.Series(self._token_pos, dtype='category'),
			'tag': pandas.Series(self._token_tag, dtype='category'),
			'prob': pandas.Series(self._token_prob, dtype='float')})

	def get_parquet_table(self):
		tokens_table_data = [
			pa.array(self._token_idx, type=pa.uint32()),
			pa.array(self._token_len, type=pa.uint8()),
			pa.array(self._token_pos, type=pa.string()),
			pa.array(self._token_tag, type=pa.string()),
			pa.array(self._token_prob, type=pa.float32())
		]

		return pa.Table.from_arrays(
			tokens_table_data,
            ['idx', 'len', 'pos', 'tag', 'prob'])

_base_path = os.path.dirname(os.path.realpath(__file__))
_cache_path = os.path.realpath(os.path.join(_base_path, '..', '..', 'data', 'cache'))

def _write_metadata(md, cache_path):
	with open(os.path.join(cache_path, "text.txt"), "r") as f:
		fulltext = '%s:%s:%s' % (md['author'], md['title'], f.read())

	import hashlib
	md['hash'] = hashlib.sha1(fulltext.encode('utf8')).hexdigest()

	with open(os.path.join(cache_path, "metadata.json"), "w") as f:
		f.write(json.dumps(md))

	return md


# custom sentence boundary detection. fixes the correct segmentation of:
#
# "Step over this line and you’ve gone beyond the law. Step over this line,
# with your massive axes and huge morningstars and heavy, heavy spiky clubs,
# and we few, we happy few who stand here with our wooden truncheons, we’ll…we’ll…
# …Well, you just better not step over the line, okay?
#
# which used to split after "we few, "
#
# see https://github.com/explosion/spaCy/blob/master/examples/pipeline/custom_sentence_segmentation.py

def can_be_sentence_start(token):
	if token.i == 0:
		return True
	elif token.nbor(-1).text == '…':
		return True
	elif token.nbor(-1).is_space:
		return can_be_sentence_start(token.nbor(-1))
	else:
		return token.nbor(-1).is_punct and token.nbor(-1).text != ','

def prevent_sentence_boundaries(doc):
	for token in doc:
		if not can_be_sentence_start(token):
			token.is_sent_start = False
	return doc

def configure_nlp(nlp):
	nlp.add_pipe(prevent_sentence_boundaries, before="parser")


class Importer:
	def __init__(self, sub_path):
		path = os.path.join(_cache_path, *sub_path)
		self._cache_path = path

	@property
	def cache_path(self):
		return self._cache_path

	@property
	def cached(self):
		return os.path.exists(self._cache_path)

	def ensure_cache(self, nlp):
		cache_path = self.cache_path
		if not os.path.exists(cache_path):
			md, locations, chunks = self._parse(nlp)
			md['parser'] = _current_parser()
			self._save(nlp, md, locations, chunks)
		return cache_path

	def _parse(self, nlp):
		raise NotImplementedError()

	def _save_table(self, df, name):
		df.to_parquet(
			os.path.join(self._cache_path, name),
			engine='fastparquet', compression="gzip")

	def _save(self, nlp, md, locations, chunks):
		# note: n_threads in nlp.pipe is currently ignored; see start.sh
		# and https://github.com/explosion/spaCy/issues/2075

		# batch_size == 1 needed for https://github.com/explosion/spaCy/issues/3607
		pipe = nlp.pipe(chunks, n_threads=1, batch_size=1)

		token_table = TokenTable()
		sentence_data = [[], [], [], [], []]

		from tqdm import tqdm
		print("creating cache for %s." %
			os.path.relpath(self._cache_path, _cache_path), flush=True)

		texts = []
		for location0, doc in tqdm(zip(locations, pipe), total=len(locations)):
			texts.append(doc.text)
			token_table.append(doc)

			for tokens in doc.sents:
				location = tuple(list(location0) + [len([t for t in tokens if not ignore_token(t)])])
				for a, value in zip(sentence_data, location):
					a.append(value)

		os.makedirs(self._cache_path, 0o777, True)

		try:
			sentence_df = pandas.DataFrame({
				'book': pandas.Series(sentence_data[0], dtype='int8'),
				'chapter': pandas.Series(sentence_data[1], dtype='int8'),
				'speaker': pandas.Series(sentence_data[2], dtype='int8'),
				'location': pandas.Series(sentence_data[3], dtype='uint16'),
				'n_tokens': pandas.Series(sentence_data[4], dtype='uint16')})

			self._save_table(sentence_df, 'sentences.parquet')

			self._save_table(token_table.get_dataframe(), 'tokens.parquet')

			with open(os.path.join(self._cache_path, "text.txt"), "w") as f:
				f.write(''.join(texts))

			_write_metadata(md, self._cache_path)

		except:
			# FIXME rm cache dir
			raise


class NovelImporter(Importer):
	# a generic importer for novel-like texts.

	_chapters = re.compile(
		r"\n\n\n\W*chapter\s+(\d+)[^\n]*\n\n", re.IGNORECASE)

	def __init__(self, path, author, novel_filename):
		novel_name = re.sub(r'\.txt$', '', novel_filename)
		self._author = author
		self._novel_name = novel_name
		super().__init__(["novels", author, novel_name])
		self._orig_text_path = os.path.join(path, author, novel_filename)

	def _parse(self, nlp):
		with open(os.path.join(self._orig_text_path), "r") as f:
			text = f.read()

		chapter_breaks = []
		expected_chapter = 1
		book = 1
		for m in NovelImporter._chapters.finditer(text):
			actual_chapter = int(m.group(1))

			if expected_chapter != actual_chapter:
				if book == 1 and expected_chapter == 2 and actual_chapter == 1:
					# we might have received "chapter 1"
					# as part of the table of contents.
					chapter_breaks = []
					expected_chapter = 1
				elif actual_chapter == 1:
					book += 1
					expected_chapter = 1
				else:
					print("bad chapter. wanted %d, got: %s" % (
						expected_chapter, m.group(0).strip()))
					chapter_breaks = []
					break

			chapter_breaks.append((book, actual_chapter, m.start(0)))
			expected_chapter += 1

		chapters = dict()
		if chapter_breaks:
			chapter_breaks.append((book, actual_chapter + 1, len(text)))
			for ((book, chapter, s), (_, _, e)) in zip(chapter_breaks, chapter_breaks[1:]):
				chapters[(book, chapter)] = text[s:e]

			first_break = chapter_breaks[0][2]
			if first_break > 0:
				chapters[(-1, -1)] = text[:first_break]
		else:
			chapters[(-1, -1)] = text

		paragraphs = []
		locations = []
		ignore_book = book <= 1

		for book, chapter in sorted(chapters.keys()):
			chapter_text = chapters[(book, chapter)]
			if ignore_book:
				book = -1

			chapter_sep = "\n\n"
			chapter_paragraphs = chapter_text.split(chapter_sep)
			chapter_paragraphs = list(map(lambda p: p + chapter_sep,
                chapter_paragraphs[:-1])) + chapter_paragraphs[-1:]
			paragraphs.extend(chapter_paragraphs)

			for j, p in enumerate(chapter_paragraphs):
				locations.append((book, chapter, -1, j))

		md = dict(
			author=self._author,
			title=self._novel_name,
			speakers={})

		return md, locations, paragraphs


class ShakespeareImporter(Importer):
	# an importer for the Shakespeare XMLs published at
	# https://github.com/severdia/PlayShakespeare.com-XML

	def __init__(self, path):
		_, filename = os.path.split(path)
		play_name = re.sub(r'\.xml$', '', filename)
		super().__init__(["shakespeare", play_name])
		self._xml_path = path

	def _parse(self, nlp):
		import xml.etree.ElementTree as ET
		from collections import defaultdict

		tree = ET.parse(self._xml_path)
		root = tree.getroot()
		speakers = defaultdict(int)
		full_speaker_names = dict()

		for persname in root.findall(".//persname"):
			full_speaker_names[persname.attrib["short"]] = persname.text

		locations = []
		texts = []

		scenes = list(root.findall(".//scene"))

		for scene_index, scene in enumerate(scenes):
			actnum = int(scene.attrib["actnum"])
			scenenum = int(scene.attrib["num"])

			for speech in scene.findall(".//speech"):
				speaker = speech.find("speaker")

				speaker_no = speakers.get(speaker.text)
				if speaker_no is None:
					speaker_no = len(speakers) + 1
					speakers[speaker.text] = speaker_no

				line_no = None
				lines = []
				for line in speech.findall("line"):
					if line.text:
						if line_no is None:
							line_no = int(line.attrib["globalnumber"])
						lines.append(line.text)

				if lines:
					locations.append((actnum, scenenum, speaker_no, line_no))
					texts.append(" ".join(lines))

		md = dict(
			author="William Shakespeare",
			title=root.find(".//title").text,
			speakers={v: full_speaker_names.get(k, k) for k, v in speakers.items()})

		return md, locations, texts


class ScreenplayImporter(Importer):
	# a generic importer for screenplays.

	def __init__(self, path, series, *where):
		self._series = series
		if len(where) > 1:
			self._episode = where[-2]
		else:
			self._episode = None

		self._filename = where[-1]
		self._title = re.sub(r'\.txt$', '', self._filename)
		self._orig_text_path = os.path.join(path, series, *where)

		super().__init__(["screenplays", series] + list(where[:-1]) + [self._title])

	def _parse(self, nlp):
		with open(self._orig_text_path, "r") as f:
			text = f.read()

		lines = []
		locations = []
		speakers = dict()
		speakers_inv = dict()

		pattern = re.compile(r'^([\w\ ]+)[^:]*:(.*)$')
		for line_no, line in enumerate(text.splitlines()):
			line = line.strip()
			speaker = None
			spoken = None

			m = pattern.match(line)
			if m:
				speaker = m.group(1).strip().upper()
				spoken = m.group(2).strip()
			elif not line.startswith('[') and not line.startswith('Original Airdate:'):
				speaker = 'stage instruction'
				spoken = line

			if speaker:
				speaker_id = speakers.get(speaker)
				if speaker_id is None:
					speaker_id = len(speakers) + 1
					speakers[speaker] = speaker_id
					speakers_inv[speaker_id] = speaker

				lines.append(spoken)
				locations.append((-1, -1, speaker_id, line_no))

		title = self._title
		if self._episode is not None:
			# incorporate series episode in title
			title = '%s, %s' % (self._episode, title)

		md = dict(
			author=self._series,
			title=title,
			speakers=speakers_inv)

		return md, locations, lines


def _create_category_importers(category, import_path):
	corpus_path = os.path.realpath(
		os.path.join(_base_path, '..', '..', 'data', 'corpus'))

	the_dir = os.path.join(corpus_path, category)
	if os.path.isdir(the_dir):
		for filename in sorted(os.listdir(the_dir)):
			if filename != 'README.md':
				for i in import_path(os.path.join(the_dir, filename)):
					yield i


def _create_importers():
	def import_flat(importer, path):
		yield importer(path)

	def import_nested(importer, depth, path):
		if not os.path.isdir(path):
			return
		for filename in sorted(os.listdir(path)):
			if filename.endswith(".txt"):
				where = []

				head = path
				for _ in range(depth):
					head, tail = os.path.split(head)
					where.append(tail)

				where = list(reversed(where))
				where.append(filename)
				yield importer(head, *where)
			else:
				p = os.path.join(path, filename)
				for i in import_nested(importer, depth + 1, p):
					yield i

	gen = (
		('shakespeare', partial(import_flat, ShakespeareImporter)),
		('novels', partial(import_nested, NovelImporter, 1)),
		('screenplays', partial(import_nested, ScreenplayImporter, 1))
	)

	for args in gen:
		for i in _create_category_importers(*args):
			yield i

def _acronym(title, length=4, ignore=set()):
	title_parts = title.upper().split(' ')
	title_parts = [s for s in title_parts if s not in ignore and s[:1].isalpha()]
	ll = [1] * len(title_parts)
	for k in range(2, length + 1):
		for j in range(len(ll) + 1):
			title = ""
			for i, p in enumerate(title_parts):
				title += p[:ll[i]]
				if len(title) >= length:
					return title[:length]
			if j < len(ll):
				ll[j] = k

	while len(title) < length:
		title += "X"

	return title[:length]

def _document_signature(md):
	author = md['author'].split(' ')[-1].upper()[:4]
	title = _acronym(md['title'], ignore=['THE', 'A', 'OF', ',', '&', 'AND', 'OR'])
	hash = md['hash'][::4].upper()
	return '%s-%s-%s' % (author, title, hash)

def _create_document(index, vocab, cache_path):
	with open(os.path.join(cache_path, "metadata.json"), "r") as f:
		md = json.loads(f.read())

	if 'hash' not in md:
		md = _write_metadata(md, cache_path)

	with open(os.path.join(cache_path, "text.txt"), "r") as f:
		text = f.read()

	sentences_table = pq.read_table(
		os.path.join(cache_path, "sentences.parquet"))
	tokens_table = pq.read_table(
		os.path.join(cache_path, "tokens.parquet"))

	return vcore.Document(index, vocab, text, sentences_table, tokens_table, md, "")

class Signatures:
	def __init__(self, kind):
		import collections
		self._kind = kind
		self._signatures = collections.OrderedDict()

	def add(self, signature, value):
		if signature in self._signatures:
			raise RuntimeError("duplicate %s signature %s" % (self._kind, signature))
		self._signatures[signature] = value

	def to_dict(self):
		return self._signatures

def _signatures(docs, parser=None):
	import hashlib
	from tqdm import tqdm

	doc_signatures = Signatures("document")

	n_sentences = 0
	for doc in docs:
		n_sentences += doc.n_sentences

	with tqdm(total=n_sentences, desc="building signatures") as progress:
		for doc in docs:
			md = doc.metadata

			if parser and 'parser' in md:
				if parser != md['parser']:
					raise RuntimeError(
						"parser mismatch in corpus cache: %s != %s." % (parser, md['parser']))

			sentence_signatures = Signatures("sentence")

			for i, s in enumerate(doc.sentences_as_text):
				s = s.strip()
				if s:
					hash = hashlib.sha1(s.encode('utf8')).hexdigest().upper()
					sentence_signature = "%s-%06d-%s" % (_acronym(s, 6), i + 1, hash[::4][:6])
					sentence_signatures.add(sentence_signature, i)

			doc_signatures.add(
				_document_signature(md),
				(doc, sentence_signatures.to_dict()))

			progress.update(doc.n_sentences)

	return doc_signatures.to_dict()

def dump_corpus(docs, dump_path):
	import os

	doc_signatures = _signatures(docs)
	for doc_sig, (doc, sentence_signatures) in doc_signatures.items():
		with open(os.path.join(dump_path, doc_sig + ".txt"), "w") as f:
			sentences = doc.sentences_as_text
			f.write("AUTHOR %s\n" % doc.metadata['author'])
			f.write("TITLE %s\n\n" % doc.metadata['title'])
			for sen_sig, sen_id in sentence_signatures.items():
				f.write("%s/%s\n: %s\n\n" % (doc_sig, sen_sig, sentences[sen_id]))

def signature_resolver(docs, parser=None):
	all_signatures = _signatures(docs, parser)

	def resolve(signatures, on_error=None):
		results = []

		if on_error is None:
			def fail_on_error(type, signature, document=None):
				raise RuntimeError("unknown %s signature %s" % (type, signature))

			on_error = fail_on_error

		for signature in signatures:
			doc_sig, sen_sig = signature.split('/')

			if doc_sig not in all_signatures:
				on_error("document", doc_sig)
				continue

			doc, sentence_signatures = all_signatures[doc_sig]

			if sen_sig not in sentence_signatures:
				on_error("sentence", sen_sig, document=doc)
				continue

			results.append((doc.id, sentence_signatures[sen_sig]))

		return results  # a list of (document id, sentence ids) pairs

	return resolve

def documents(vocab, nlp):
	importers = list(_create_importers())

	if not all(importer.cached for importer in importers):
		print("caching documents.", flush=True)

		for importer in importers:
			if not importer.cached:
				importer.ensure_cache(nlp)

	from tqdm import tqdm
	docs = []
	for i, importer in enumerate(tqdm(importers, desc="importing documents")):
		docs.append(_create_document(i, vocab, importer.ensure_cache(nlp)))
	return docs


if __name__ == "__main__":
	base_path = os.path.dirname(os.path.realpath(__file__))
	importer = NovelImporter(os.path.join(
		base_path, '..', '..', 'data', 'corpus', 'novels',
		'Charles Dickens', "A Child's Dream of a Star"))

	nlp = spacy.load("en_core_web_lg")
	configure_nlp(nlp)
	importer(nlp)
