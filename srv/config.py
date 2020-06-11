import os
import json

from pathlib import Path


class Config:
	def __init__(self):
		# hard-coded default configuration.
		self._config = dict(
			port=8080,
			password=None,
			cookie_secret='very.sikrit.',
			deploy_url=None,
			metrics=[],
			embeddings=['fasttext'],
			fasttext="wiki-news-300d-1M-subword.vec")

		self._script_dir = Path(os.path.dirname(os.path.realpath(__file__)))

		config_path = self._script_dir.parent / ".config.json"
		if config_path.exists():
			with open(config_path, "r") as f:
				self._config = json.loads(f.read())

	@property
	def port(self):
		return int(self._config["port"])

	@property
	def password(self):
		return self._config["password"]

	@property
	def cookie_secret(self):
		return self._config["cookie_secret"]

	@property
	def deploy_url(self):
		return self._config["deploy_url"]

	@property
	def metrics(self):
		return self._config["metrics"]  # ['apsynp', 'nicdm', 'percentiles']

	@property
	def embeddings(self):
		return self._config["embeddings"]  # ['wn2vec', 'elmo']

	@property
	def fasttext(self):
		return self._config["fasttext"]
