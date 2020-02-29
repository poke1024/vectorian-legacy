![Vectorian Screenshot](/docs/screenshot.png)

Vectorian is a high-performance search engine for intertextual references powered by
<a href="https://fasttext.cc/">fastText</a>, <a href="https://spacy.io/">spaCy</a> and
<a href="https://github.com/poke1024/simileco">simileco</a>.

# Minimal Installation

Installation should work on macOS and Linux. On Windows you should use Docker or a VM.

## Python Packages

```
conda create --name vectorian python=3.7
conda activate vectorian

git clone https://github.com/poke1024/vectorian
git checkout v2
cd vectorian

# install core dependencies
pip install -r requirements.txt
```

## Necessary Data Files and Additional Dependencies

### spaCy en_core_web_lg

```
# download spaCy's large English model
conda activate vectorian
python -m spacy download en_core_web_lg
```

### fastText

Download `wiki-news-300d-1M-subword` from https://fasttext.cc/docs/en/english-vectors.html.

Unzip and put this to `/path/to/vectorian/data/fasttext/wiki-news-300d-1M-subword.vec`.

There's also support for the larger crawl-300d-2M-subword. Note that this is not recommeneded for
standard installations, as loading and preprocessing times are high.

### Installing the Eigen library

Eigen is needed by vectorian's C++ backend and by <a href="https://github.com/poke1024/simileco">simileco</a>.

#### on macOS

Use `brew install eigen`.

On some versions of macOS, you might need to patch eigen:

https://stackoverflow.com/questions/46356153/xcode-9-falls-to-build-partial-template-specialization-in-c

#### on Ubuntu

```
sudo apt install libeigen3-dev
```

# Adding Text Data to the Vectorian

Text data lives inside Vectorian's `data/corpus` folder. You add files
and Vectorian will preprocess and load them automatically on startup.

However you need to adhere to a given structure of three types of files
that Vectorian needs to preprocess files in an optimal way.

Files are accordingly organized into three subfolders:

* `data/corpus/shakespeare`: receives XML shakespeare files. The files have
to be in the format used by <a href="https://github.com/severdia/PlayShakespeare.com-XML">playshakespeare.com</a>
* `data/corpus/nodels`: contains folders of authors and in these folders
plain text files of the author's novels.
* `data/corpus/screenplays`: contains screenplays.

Not all these folders have to exist, you can, for example, just add novels.

Here's an example layout:

```
corpus
	novels
		Charles Dickens
			Hard Times.txt
			The Pickwick Papers.txt
		Jane Austen
			Northanger Abbey.txt
screenplays
	that_exciting_series
		pilot.txt
		series1
			season1.txt
shakespeare
	ps_hamlet.xml
	ps_henry_v.xml
```

# Using Vectorian

## Launching Vectorian

```
conda activate vectorian

cd /path/to/vectorian
python ./srv/main.py
```

After starting up, Vectorian should be available at `http://localhost:8080/`.

# Developer Instructions

## Troubleshooting

### general

If you see `'Eigen/Core' file not found` during startup, it means
that the Eigen library has not been installed properly (or is not
in your PATH). You can configure custom paths via `srv/cpp/vcore.cpp`.

### on macOS

On macOS, if you observe strange crashes related to numba or llvm,
you might need to do (see https://github.com/numba/numba/issues/4256):

`pip install pyarrow==0.12.1`

Also see: https://www.mail-archive.com/dev@arrow.apache.org/msg13667.html

### on Ubuntu

Under GCC, there might problems with ABI compatibility of arrow and
pyarrow libs (see https://arrow.apache.org/docs/python/development.html).
`_GLIBCXX_USE_CXX11_ABI` can help.

## Building elm modules for frontend

`web/build.sh`

## Manually building the C++ component

```
c++ -O3 -larrow -Wall -shared -std=c++17 -I/usr/include/eigen3/ -fPIC `python3 -m pybind11 --includes` src.cpp -o vcore`python3-config --extension-suffix`
```

## Debugging the C++ component

on macOS:
```
export DEBUG_VECTORIAN=1
lldb python -- vectorian/srv/main.py
```

on Linux:
```
export DEBUG_VECTORIAN=1
export ASAN_OPTIONS=verify_asan_link_order=0
gdb --args python vectorian/srv/main.py
```

## Running Vectorian as a systemd service

Here's a template for Vectorian as a systemd service:

```
[Unit]
Description=The Vectorian
After=multi-user.target

[Service]
Type=simple
ExecStart=/your/python3 your/vectorian/srv/main.py
WorkingDirectory=your/vectorian
Restart=always
RestartSec=10
PrivateTmp=true
StandardOutput=syslog
StandardError=syslog
Environment=OPENBLAS_NUM_THREADS=2
Environment=VECTORIAN_PORT=8080

[Install]
WantedBy=multi-user.target
```

Install this as `vectorian.service` into `/etc/systemd/system/vectorian.service`.

Now you can use these useful commands:

```
systemctl daemon-reload

systemctl start vectorian.service
systemctl status vectorian.service

tail -f /var/log/syslog
```
