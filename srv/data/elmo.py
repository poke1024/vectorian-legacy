import os
import numpy
import sklearn
import cpp.vcore as vcore
import data.corpus
import multiprocessing

_query_embedder = None


def create_embedder():
	print("creating elmo embedder.", flush=True)

	basepath = os.path.join(os.path.dirname(
		os.path.realpath(__file__)), "..", "..", "data", "elmo")
	from allennlp.commands.elmo import ElmoEmbedder

	return ElmoEmbedder(
		options_file=os.path.join(basepath, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'),
		weight_file=os.path.join(basepath, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'))


def query(sentence):
	global _query_embedder

	if _query_embedder is None:
		_query_embedder = create_embedder()

	# pre-warm elmo for the given sentence.
	for _ in range(7):
		_query_embedder.embed_sentence(sentence)

	vectors = _query_embedder.embed_sentence(sentence)
	vectors = numpy.average(vectors, axis=0)
	vectors = sklearn.preprocessing.normalize(vectors)

	return vectors


def _compute_elmo(args):
	cache_path, sentences, n_tokens = args

	elmo_path = os.path.join(cache_path, "elmo.dat")
	if os.path.exists(elmo_path):
		return

	print("computing %s" % elmo_path, flush=True)

	matrix = vcore.LargeMatrix(elmo_path + ".tmp")
	matrix.create(n_tokens, 1024)

	embedder = create_embedder()
	try:
		k = 0

		from tqdm import tqdm

		for sentence in tqdm(sentences):
			if len(sentence) == 0:
				continue

			vectors = embedder.embed_sentence(sentence)
			vectors = numpy.average(vectors, axis=0)
			vectors = sklearn.preprocessing.normalize(vectors)

			assert len(vectors) == len(sentence)
			for i in range(len(sentence)):
				matrix.write(k + i, vectors[i])
			k += len(sentence)
	except:
		matrix.close()
		os.unlink(elmo_path + ".tmp")
	finally:
		matrix.close()
		del embedder

	os.rename(elmo_path + ".tmp", elmo_path)


def precompute(n_processes=2):
	print("loading spacy... ", flush=True, end="")
	import spacy
	print("done.", flush=True)

	nlp = spacy.load("en_core_web_lg")

	vocab = vcore.Vocabulary()
	documents = data.corpus.documents(vocab, nlp)
	arguments = [(document.path, document.sentences, document.n_tokens) for document in documents]

	if n_processes == 1:
		for args in arguments:
			_compute_elmo(args)
	else:
		pool = multiprocessing.Pool(n_processes)
		pool.map(_compute_elmo, arguments)
