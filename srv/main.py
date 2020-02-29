import tornado.ioloop
import tornado.web
import tornado.platform.asyncio
from tornado import websocket

import json
import socket
import locale
import asyncio
import janus
import traceback

from config import Config


class BaseHandler(tornado.web.RequestHandler):
	def get_current_user(self):
		return self.get_secure_cookie("user")


class LoginHandler(BaseHandler):
	def __init__(self, config):
		self._config = config

	def get(self):
		self.render("login.html")

	def post(self):
		if self.get_argument("password") == self._config.password:
			self.set_secure_cookie("user", "researcher")
			self.redirect("/")
		else:
			self.redirect("/login")


class MainHandler(BaseHandler):
	def initialize(self, config):
		if config.password is None:
			self.set_secure_cookie("user", "researcher")

	@tornado.web.authenticated
	def get(self):
		self.render("main.html")


class SocketHandler(websocket.WebSocketHandler):
	def initialize(self, app):
		self._app = app

		self._delegate = self._app.start_session(self._ws_send)
		self._delegate_proxy = self._delegate.proxy()

		loop = asyncio.get_event_loop()
		self._send_queue = janus.Queue(loop=loop)
		self._send_task = None

	def check_origin(self, origin):
		return True

	def _ws_send(self, message):
		self._send_queue.sync_q.put(message)

	async def _send_task_handler(self):
		while True:
			message = await self._send_queue.async_q.get()
			await self.write_message(message)

	def open(self):
		print("SocketHandler.open")
		if self.get_secure_cookie("user").decode("utf8") != "researcher":
			raise RuntimeError("denied illegal access")

		self._send_task = asyncio.ensure_future(
			self._send_task_handler())
		print("SocketHandler.open done")

	def on_message(self, message):
		print("SocketHandler.on_message", message)
		decoded = json.loads(message)
		self._delegate_proxy.on_ws_receive(decoded)

	def on_close(self):
		print("SocketHandler.on_close")
		if self._send_task:
			self._send_task.cancel()
			self._send_task = None

		# important to cleanup and stop delegate, otherwise there
		# will be a dangling thread.

		if self._delegate:
			self._delegate.stop()
			self._delegate = None


class BatchResultHandler(BaseHandler):
	def initialize(self, app):
		self._app = app


class BatchHandler(BaseHandler):
	def initialize(self, app):
		self._app = app

	def _render_tqdm(self, n, total, rate, elapsed, **kwargs):
		if n >= total:
			self.render(
				"info.html",
				info="Batch completed.")
		else:
			import datetime
			import math

			self.render(
				"batch.html",
				progress_value=n,
				progress_max=total,
				progress_text="%d" % math.floor(100 * n / total),
				elapsed=str(datetime.timedelta(seconds=math.floor(elapsed))),
				remaining=str(datetime.timedelta(seconds=math.floor((total - n) / rate))) if rate else "",
				rate="%.1f / s" % rate if rate else "")

	async def get(self):
		import evaluation
		evaluation = evaluation.Evaluation.read()

		evaluator = self._app.evaluator
		if evaluator:
			tqdm = await evaluator.tqdm()
			self._render_tqdm(**tqdm.format_dict)
		else:
			self.render(
				"info.html",
				info="No batch running.",
				body=evaluation.optimal() if evaluation else "")


def make_app(app, config):
	tornado_path = os.path.realpath(os.path.join(
		os.path.dirname(os.path.realpath(__file__)), "tornado"))

	handlers = [
		(r'/static/(.*)',
			tornado.web.StaticFileHandler,
			{'path': os.path.join(tornado_path, 'static')}),
		(r"/", MainHandler, dict(config=config)),
		(r"/login", LoginHandler, dict(config=config)),
		(r'/ws', SocketHandler, dict(app=app)),
		(r'/batch', BatchHandler, dict(app=app)),
	]

	application = tornado.web.Application(
		handlers,
		template_path=os.path.join(tornado_path, 'templates'),
		cookie_secret=config.cookie_secret,
		login_url='/' if config.password is None else '/login',
		xsrf_cookies=True)

	if config.deploy_url is not None:
		application.add_handlers(config.deploy_url, handlers)

	return application


if __name__ == "__main__":
	config = Config()

	# you might need to install via "dpkg-reconfigure locales" on Debian.
	locale.setlocale(
		locale.LC_ALL,
		'en_US.utf8' if (config.deploy_url is not None) else 'en_US')

	import os
	import platform

	if platform.system() == 'Darwin':
		port = 8080
	else:
		if 'VECTORIAN_PORT' not in os.environ:
			raise RuntimeError("missing VECTORIAN_PORT env var.")
		port = int(os.environ['VECTORIAN_PORT'])

	from app import App

	try:
		app = make_app(App(), config)
		print("running on port %d." % port)
		app.listen(port)
		tornado.ioloop.IOLoop.current().start()
	except Exception as e:
		print("tornado aborts with an exception.", flush=True)
		traceback.print_exc()
