import os

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.realpath(os.path.join(base_path, '..', 'data'))

    os.makedirs(data_path, 0o777, True)

    for category in ('novels', 'screenplays', 'shakespeare'):
        os.makedirs(os.path.join(data_path, 'corpus', category), 0o777, True)

    os.makedirs(os.path.join(data_path, 'fasttext'), 0o777, True)
    os.makedirs(os.path.join(data_path, 'wn2vec'), 0o777, True)
    os.makedirs(os.path.join(data_path, 'elmo'), 0o777, True)

    #import urllib.request
    #urllib.request.urlretrieve(
    #    "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M-subword.vec.zip",
    #    os.path.join(data_path, 'fasttext', 'fasttext.zip'))
