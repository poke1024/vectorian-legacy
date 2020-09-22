#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

// mini docs.
// - a (partial) Metric is generated once for each query and
// is used for various documents; for fast embeddings, this
// will hold the computed similarity matrix.
// - a Matcher is generated once for each document and
// operates on a specific pair of (query, document)

namespace py = pybind11;

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::MatrixXf;
using Eigen::ArrayXf;

#include <iostream>
#include <fstream>
#include <functional>
#include <queue>
#include <mutex>
#include <map>
#include <set>
#include <chrono>

#include <sys/mman.h>
#include <fcntl.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/python/pyarrow.h>

#define PPK_ASSERT_ENABLED 1
#include "ppk_assert.h"

#include "aligner.h"


typedef int32_t token_t;

typedef Eigen::Array<token_t, Eigen::Dynamic, 1> TokenIdArray;

class Embedding;

typedef std::shared_ptr<Embedding> EmbeddingRef;

typedef std::unordered_map<int, float> POSWMap;

const float smith_waterman_zero = 0.5;


std::shared_ptr<arrow::Table> unwrap_table(PyObject *p_table) {
	arrow::Result<std::shared_ptr<arrow::Table>> table(
        arrow::py::unwrap_table(p_table));
	if (!table.ok()) {
    	std::ostringstream err;
    	err << "PyObject of type " << Py_TYPE(p_table)->tp_name <<
    	    " could not get converted to a pyarrow table";
		throw std::runtime_error(err.str());
	}
	return *table;
}

std::shared_ptr<arrow::Table> unwrap_table(const py::object &p_table) {
    return unwrap_table(p_table.ptr());
}

#if PYARROW_0_12_1
auto column_data(const std::shared_ptr<arrow::Column> &c) {
	return c->data();
}
#else
auto column_data(const std::shared_ptr<arrow::ChunkedArray> &c) {
	return c;
}
#endif

template<typename ArrowType, typename Array>
static void ensure_type(const Array &array) {
    if (array->type_id() != arrow::TypeTraits<ArrowType>::type_singleton()->id()) {
        std::string got = array->type()->name();
        std::string expected = arrow::TypeTraits<ArrowType>::type_singleton()->name();

        std::ostringstream err;
        err << "parquet data type of chunk is wrong. expected " <<
            expected << ", got " + got + ".";

        throw std::runtime_error(err.str());
    }
}

template<typename ArrowType, typename CType, typename F>
void for_each_column(const std::shared_ptr<arrow::Table> &p_table, const F &p_f, int p_first_column = 0) {
		for (int i = p_first_column; i < p_table->num_columns(); i++) {
			const auto data = column_data(p_table->column(i));
			size_t offset = 0;

		    for (int64_t k = 0; k < data->num_chunks(); k++) {
                auto array = data->chunk(k);
                ensure_type<ArrowType>(array);

                auto num_array = std::static_pointer_cast<arrow::NumericArray<ArrowType>>(array);

                const Eigen::Map<Eigen::Array<CType, Eigen::Dynamic, 1>> v(
                    const_cast<CType*>(num_array->raw_values()), num_array->length());

                p_f(i, v, offset);

                offset += num_array->length();
		    }
		}
}

class LargeMatrix {
	#pragma pack(push, 1)
	struct Header {
		int64_t magic;
		int64_t width;
		int64_t height;
	};
	#pragma pack(pop)

	enum {
		MAGIC = 0xDADA
	};

	const std::string m_path;

	int m_fd;

	size_t m_width;
	size_t m_height;
	size_t m_size;

	void *m_map;
	float *m_data;

public:
	typedef Eigen::Map<const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> View;

	LargeMatrix(const std::string &p_path) :
		m_path(p_path),
		m_fd(-1),
		m_map(nullptr),
		m_data(nullptr) {
	}

	~LargeMatrix() {
		close();
	}

	LargeMatrix(const LargeMatrix&) = delete;

	bool exists() const {
		struct stat buffer;
		return ::stat(m_path.c_str(), &buffer) == 0;
	}

	void close() {
		if (m_fd != -1) {
			const int msync_res = msync(m_map, m_size, MS_SYNC);

			if (munmap(m_map, m_size) == -1) {
					::close(m_fd);
					m_fd = -1;
					m_map = nullptr;
					m_data = nullptr;
					throw std::runtime_error("could not munmap file at " + m_path);
			}

			::close(m_fd);
			m_fd = -1;
			m_map = nullptr;
			m_data = nullptr;

			if (msync_res == -1) {
					throw std::runtime_error("could not msync file at " + m_path);
			}
		}
	}

	void write(size_t p_index, py::array_t<double> p_data) {
		if (m_fd == -1) {
			throw std::runtime_error("not open");
		}

		auto buffer = p_data.request();

		if (buffer.ndim != 1) {
			std::ostringstream err;
			err << "wrong numpy array dimension ";
			err << buffer.ndim;
			throw std::runtime_error(err.str());
		}
		if (size_t(buffer.shape[0]) != m_width) {
			std::ostringstream err;
			err << "wrong numpy array length ";
			err << buffer.shape[0] << " != " << m_width;
			throw std::runtime_error(err.str());
		}
		if (p_index < 0 || p_index >= m_height) {
			std::ostringstream err;
			err << "illegal index " << p_index << " not in [0, " << m_height << "[";
			throw std::runtime_error(err.str());
		}

		const double *input = static_cast<double*>(buffer.ptr);
		float *row = m_data;
		row += m_width * p_index;
		for (size_t i = 0; i < m_width; i++) {
			row[i] = input[i];
		}
	}

	View view(size_t p_row0, size_t p_rows) const {
		if (m_fd == -1) {
			throw std::runtime_error("not open");
		}
		if (p_row0 + p_rows > m_height) {
			throw std::runtime_error("illegal access in Matrix");
		}

		return View(
			m_data + m_width * p_row0,
			p_rows,
			m_width);
	}

	void open() {
		if (m_fd != -1) {
			return;
		}

		const int fd = ::open(m_path.c_str(), O_RDONLY, 0);
		if (fd == -1) {
				throw std::runtime_error("could not open file " + m_path);
		}

		Header header;
		if (::read(fd, &header, sizeof(header)) != sizeof(header)) {
				::close(fd);
				throw std::runtime_error("could not read header for file at " + m_path);
		}

		if (header.magic != MAGIC) {
			throw std::runtime_error("Matrix header is damaged");
		}

		const size_t w = header.width;
		const size_t h = header.height;
		const size_t size = h * w * sizeof(float) + sizeof(Header);

		// might use MAP_POPULATE here.
		void *map = mmap(0, size, PROT_READ, MAP_PRIVATE, fd, 0);
		if (map == MAP_FAILED) {
			::close(fd);
			throw std::runtime_error("could not mmap file at " + m_path);
		}

		m_data = reinterpret_cast<float*>(
			static_cast<char*>(map) + sizeof(Header));

		m_fd = fd;
		m_map = map;
		m_size = size;

		m_width = w;
		m_height = h;
	}

	void create(size_t p_h, size_t p_w) {
		if (m_fd != -1) {
			throw std::runtime_error("already open");
		}

		const size_t size = p_h * p_w * sizeof(float) + sizeof(Header);

		const int fd = ::open(m_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
		if (fd == -1) {
				throw std::runtime_error("could not open file " + m_path);
		}

		// make file the requested size.
		if (::lseek(fd, size - 1, SEEK_SET) == -1) {
				::close(fd);
				throw std::runtime_error("could not resize file at " + m_path);
		}
		if (::write(fd, "", 1) == -1) {
				::close(fd);
				throw std::runtime_error("could not resize file at " + m_path);
		}

		void *map = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if (map == MAP_FAILED) {
			::close(fd);
			throw std::runtime_error("could not mmap file at " + m_path);
		}

		Header *header = static_cast<Header*>(map);
		header->magic = MAGIC;
		header->width = p_w;
		header->height = p_h;

		m_data = reinterpret_cast<float*>(
			static_cast<char*>(map) + sizeof(Header));

		m_fd = fd;
		m_map = map;
		m_size = size;

		m_width = p_w;
		m_height = p_h;
	}
};

typedef std::shared_ptr<LargeMatrix> LargeMatrixRef;

class Matcher;
typedef std::shared_ptr<Matcher> MatcherRef;
class Query;
typedef std::shared_ptr<Query> QueryRef;
class Document;
typedef std::shared_ptr<Document> DocumentRef;
class ResultSet;
typedef std::shared_ptr<ResultSet> ResultSetRef;

class Metric : public std::enable_shared_from_this<Metric> {
	// a point could be made that this is only a partial
	// metric, since it's only between corpus and needle
	// (and not between corpus and corpus), but let's
	// keep names simple.

public:
	virtual ~Metric() {
	}

	virtual MatcherRef create_matcher(
		const QueryRef &p_query,
		const DocumentRef &p_document) = 0;

	virtual const std::string &name() const = 0;

	virtual const std::string &origin(int p_token_id_s, int p_query_token_index) const {
		return name();
	}
};

typedef std::shared_ptr<Metric> MetricRef;

class FastMetric : public Metric {
protected:
	MatrixXf m_similarity;
	const EmbeddingRef m_embedding;
	const float m_pos_mismatch_penalty;
	const float m_similarity_falloff;
	const float m_similarity_threshold;
	const POSWMap m_pos_weights;

public:
	FastMetric(
			const EmbeddingRef &p_embedding,
			float p_pos_mismatch_penalty,
			float p_similarity_falloff,
			float p_similarity_threshold,
			const POSWMap &p_pos_weights) :

			m_embedding(p_embedding),
			m_pos_mismatch_penalty(p_pos_mismatch_penalty),
			m_similarity_falloff(p_similarity_falloff),
			m_similarity_threshold(p_similarity_threshold),
			m_pos_weights(p_pos_weights) {
	}

	inline MatrixXf &w_similarity() {
		return m_similarity;
	}

	inline const MatrixXf &similarity() const {
		return m_similarity;
	}

	inline float pos_mismatch_penalty() const {
		return m_pos_mismatch_penalty;
	}

	inline float similarity_threshold() const {
		return m_similarity_threshold;
	}

	inline float similarity_falloff() const {
		return m_similarity_falloff;
	}

	inline const POSWMap &pos_weights() const {
		return m_pos_weights;
	}

	inline float pos_weight(int tag) const {
		const auto w = m_pos_weights.find(tag);
		if (w != m_pos_weights.end()) {
			return w->second;
		} else {
			return 1.0f;
		}
	}

	virtual MatcherRef create_matcher(
		const QueryRef &p_query,
		const DocumentRef &p_document);

	virtual const std::string &name() const;
};

typedef std::shared_ptr<FastMetric> FastMetricRef;

class CompositeMetric : public FastMetric {
	const FastMetricRef m_a;
	const FastMetricRef m_b;
	std::string m_name;
	Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> m_is_from_a;

public:
	CompositeMetric(
		const MetricRef &p_a,
		const MetricRef &p_b,
		float t) :

		FastMetric(
			EmbeddingRef(),
			std::dynamic_pointer_cast<FastMetric>(p_a)->pos_mismatch_penalty(),
			std::dynamic_pointer_cast<FastMetric>(p_a)->similarity_falloff(),
			std::dynamic_pointer_cast<FastMetric>(p_a)->similarity_threshold(),
			std::dynamic_pointer_cast<FastMetric>(p_a)->pos_weights()),
		m_a(std::dynamic_pointer_cast<FastMetric>(p_a)),
		m_b(std::dynamic_pointer_cast<FastMetric>(p_b)) {

		const float t2 = 2.0f * t; // [0, 2] for linear interpolation
		const float k = 1.0f + std::abs(t2 - 1.0f); // [1, 2] for normalization

		const MatrixXf arg_a = m_a->similarity() * ((2.0f - t2) / k);
		m_similarity = arg_a.cwiseMax(m_b->similarity() * (t2 / k));

		m_is_from_a = m_similarity.cwiseEqual(arg_a);

		if (t == 0.0f) {
			m_name = m_a->name();
		} else if (t == 1.0f) {
			m_name = m_b->name();
		} else {
			char buf[32];
			snprintf(buf, 32, "%.2f", t);
			m_name = m_a->name() + " + " + m_b->name() + " @" + buf;
		}
	}

	virtual const std::string &name() const {
		return m_name;
	}

	virtual const std::string &origin(int p_token_id_s, int p_query_token_index) const {
		return m_is_from_a(p_token_id_s, p_query_token_index) ?  m_a->name() :  m_b->name();
	}
};

struct Location {
	int8_t book;
	int8_t chapter;
	int8_t speaker;
	int16_t paragraph;
};

struct Sentence : public Location {
	int16_t n_tokens;
	int32_t token_at;
};

struct Token {
	token_t id;
	int32_t idx;
	int8_t len;
	int8_t pos; // universal POS tags
	int8_t tag; // Penn TreeBank style POS tags
};

class FastSentenceScores {
private:
	const FastMetricRef m_metric;
	const Token * const s_tokens;
	const int32_t _s_len;
	const Token * const t_tokens;

public:
	inline FastSentenceScores(
		const FastMetricRef &metric,
		const Token * const s_tokens,
		const int32_t s_len,
		const Token * const t_tokens) :

		m_metric(metric),
		s_tokens(s_tokens),
		_s_len(s_len),
		t_tokens(t_tokens) {
	}

	inline int32_t s_len() const {
	    return _s_len;
	}

	inline float similarity(int i, int j) const {

		const Token &s = s_tokens[i];
		const Token &t = t_tokens[j];
		float score;

		if (s.id == t.id) {
			score = 1.0f;
		} else {
			const auto &sim = m_metric->similarity();
			score = sim(s.id, j);
		}

		return score;
	}

	inline float weight(int i, int j) const {

		const Token &s = s_tokens[i];
		const Token &t = t_tokens[j];

		// weight based on PennTree POS tag.
		float weight = m_metric->pos_weight(t.tag);

		// difference based on universal POS tag. do not apply
		// if the token is the same, but only POS is different,
		// since often this will an error in the POS tagging.
		if (s.pos != t.pos && s.id != t.id) {
			weight *= 1.0f - m_metric->pos_mismatch_penalty();
		}

		return weight;
	}

	inline float operator()(int i, int j) const {

		float score = similarity(i, j) * weight(i, j);

		if (score <= m_metric->similarity_threshold()) {
			score = 0.0f;
		}

		return score;
	}
};

struct WordVectors {
	typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> V;
	typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> R;

	V raw;
	V normalized;
	R apsynp;
	V neighborhood;

	void update_normalized() {
		normalized.resize(raw.rows(), raw.cols());
		for (Eigen::Index j = 0; j < raw.rows(); j++) {
			const float len = raw.row(j).norm();
			normalized.row(j) = raw.row(j) / len;
		}
	}
};

class ElmoMetric : public Metric {
	const std::string m_name;
	std::vector<WordVectors> m_embeddings;

public:
	ElmoMetric(
		const std::string &p_needle_text,
		const std::vector<Token> &p_needle) :
		m_name("elmo") {

		py::list tokens;
		for (const Token &t : p_needle) {
			tokens.append(py::str(p_needle_text.substr(t.idx, t.len)));
		}

		const int numSamples = 1;
		for (int i = 0; i < numSamples; i++) {
				auto elmo = py::module::import("data.elmo");
				py::array_t<double> embeddings = elmo.attr("query")(tokens);

				auto buffer = embeddings.request();
				if (buffer.ndim != 2) {
					std::ostringstream err;
					err << "wrong elmo embedding array dimension ";
					err << buffer.ndim;
					throw std::runtime_error(err.str());
				}

				m_embeddings.emplace_back(WordVectors());
				WordVectors &v = m_embeddings.back();

				const size_t h = buffer.shape[0];
				const size_t w = buffer.shape[1];

				v.raw.resize(h, w);
				const double *data = static_cast<double*>(buffer.ptr);
				for (size_t i = 0; i < h; i++) {
					for (size_t j = 0; j < w; j++) {
						v.raw(i, j) = *data++;
					}
				}

				v.update_normalized();
		}
	}

	virtual MatcherRef create_matcher(
		const QueryRef &p_query,
		const DocumentRef &p_document);

	virtual const std::string &name() const {
		return m_name;
	}

	inline size_t num_samples() const {
		return m_embeddings.size();
	}

	inline const WordVectors &t_embeddings(size_t p_sample) const {
		return m_embeddings.at(p_sample);
	}
};

typedef std::shared_ptr<ElmoMetric> ElmoMetricRef;

class FastScores {
	const QueryRef &m_query;
	const DocumentRef &m_document;
	const FastMetricRef m_metric;

	mutable std::vector<Token> m_filtered;

public:
	FastScores(
		const QueryRef &p_query,
		const DocumentRef &p_document,
		const FastMetricRef &p_metric);

	inline FastSentenceScores create_sentence_scores(
		size_t p_s_offset,
		size_t p_s_len,
		int p_pos_filter) const;

	inline bool good() const {
		return true;
	}

	inline int variant() const {
		return 0;
	}
};

class ElmoSentenceScores {
private:
	const LargeMatrix::View s_embeddings;
	const WordVectors &t_embeddings;

public:
	inline ElmoSentenceScores(
		const LargeMatrix::View &s_embeddings,
		const WordVectors &t_embeddings) :

		s_embeddings(s_embeddings),
		t_embeddings(t_embeddings) {
	}

	inline int s_len() const {
	    PPK_ASSERT(false);
	    return 0;
	}

	inline float similarity(int i, int j) const {
		return s_embeddings.row(i).dot(t_embeddings.raw.row(j));
	}

	inline float weight(int i, int j) const {
		return 1.0f;
	}

	inline float operator()(int i, int j) const {
		return similarity(i, j);
	}
};

class ElmoScores {
	const DocumentRef m_document;
	const ElmoMetricRef m_metric;
	const int m_sample;
	LargeMatrixRef m_document_matrix;
	bool m_good;

public:
	ElmoScores(
		const DocumentRef &p_document,
		const ElmoMetricRef &p_metric,
		const int p_sample);

	inline ElmoSentenceScores create_sentence_scores(
		size_t p_s_offset,
		size_t p_s_len,
		int p_pos_filter) const {

		PPK_ASSERT(m_good);
		PPK_ASSERT(p_pos_filter < 0);

		return ElmoSentenceScores(
			m_document_matrix->view(p_s_offset, p_s_len),
			m_metric->t_embeddings(m_sample));
	}

	inline bool good() const {
		return m_good;
	}

	inline int variant() const {
		return m_sample;
	}
};

struct WeightedIDF {
	const std::vector<float> *idf;
	float w;
};

class EmbeddingSimilarity {
public:
	virtual ~EmbeddingSimilarity() {
	}

	virtual void build_matrix(
		const WordVectors &p_embeddings,
		const TokenIdArray &p_a,
		const TokenIdArray &p_b,
		const WeightedIDF *p_a_prob,
		MatrixXf &r_matrix) const = 0;

	virtual void load_percentiles(const std::string &p_path, const std::string &p_name) {
	}
};

typedef std::shared_ptr<EmbeddingSimilarity> EmbeddingSimilarityRef;


class Embedding : public std::enable_shared_from_this<Embedding> {
	const std::string m_name;

public:
	Embedding(const std::string &p_name) : m_name(p_name) {
	}

	virtual ~Embedding() {
	}

	virtual token_t lookup(const std::string &p_token) const {
		return -1;
	}

	virtual MetricRef create_metric(
		const TokenIdArray &p_vocabulary_to_embedding,
		const std::vector<float> &p_idf,
		const std::string &p_embedding_similarity,
		const std::string &p_needle_text,
		const std::vector<Token> &p_needle,
		float p_pos_mismatch_penalty,
		float p_similarity_falloff,
		float p_similarity_threshold,
		const POSWMap &p_pos_weights,
		const float p_idf_weight) = 0;

	const std::string &name() const {
		return m_name;
	}
};


class ElmoEmbedding : public Embedding {
public:
	ElmoEmbedding() : Embedding("elmo") {
	}

	virtual MetricRef create_metric(
		const TokenIdArray &p_vocabulary_to_embedding,
		const std::vector<float> &p_idf,
		const std::string &p_embedding_similarity,
		const std::string &p_needle_text,
		const std::vector<Token> &p_needle,
		float p_pos_mismatch_penalty,
		float p_similarity_falloff,
		float p_similarity_threshold,
		const POSWMap &p_pos_weights,
		const float p_idf_weight);
};

typedef std::shared_ptr<ElmoEmbedding> ElmoEmbeddingRef;


const std::string &FastMetric::name() const {
	return m_embedding->name();
}


template<typename Distance>
class SimilarityMeasure : public EmbeddingSimilarity {
protected:
		Distance m_distance;

public:
	SimilarityMeasure(
		const WordVectors &p_vectors,
		const Distance p_distance = Distance()) : m_distance(p_distance) {
	}

	virtual void build_matrix(
		const WordVectors &p_embeddings,
		const TokenIdArray &p_a,
		const TokenIdArray &p_b,
		const WeightedIDF *p_a_idf,
		MatrixXf &r_matrix) const {

		const size_t n = p_a.rows();
		const size_t m = p_b.rows();
		r_matrix.resize(n, m);

		const float prob_w = p_a_idf ? p_a_idf->w : 0.0f;
		float max_w = 0.0f;

		for (size_t i = 0; i < n; i++) { // e.g. for each token in Vocabulary
			const token_t s = p_a[i];

			if (s >= 0) {
				for (size_t j = 0; j < m; j++) { // e.g. for each token in needle

					const token_t t = p_b[j];
					r_matrix(i, j) = (t >= 0) ? m_distance(p_embeddings, s, t) : 0.0f;
				}

				if (prob_w > 0.0f) {
					float w =  std::pow(p_a_idf->idf->at(i), prob_w);
					max_w = std::max(w, max_w);
					r_matrix.row(i) *= w;
				}

			} else { // token in Vocabulary, but not in Embedding

				for (size_t j = 0; j < m; j++) {
					r_matrix(i, j) = 0.0f;
				}
			}
		}

		if (max_w > 0.0f) {
			r_matrix /= max_w;
		}
	}

};

class MaximumSimilarityMeasure : public EmbeddingSimilarity {
private:
	std::vector<EmbeddingSimilarityRef> m_measures;

public:
	MaximumSimilarityMeasure(
		const std::vector<EmbeddingSimilarityRef> &p_measures) : m_measures(p_measures) {
	}

	virtual void build_matrix(
		const WordVectors &p_embeddings,
		const TokenIdArray &p_a,
		const TokenIdArray &p_b,
		const WeightedIDF *p_a_prob,
		MatrixXf &r_matrix) const {

		if (m_measures.empty()) {
			r_matrix.setZero();
		} else {
			MatrixXf temp_matrix;
			temp_matrix.resize(r_matrix.rows(), r_matrix.cols());

			for (size_t i = 0; i < m_measures.size(); i++) {
				MatrixXf &target = (i == 0) ? r_matrix : temp_matrix;
				m_measures[i]->build_matrix(
					p_embeddings, p_a, p_b, p_a_prob, target);
				if (i > 0) {
					r_matrix = temp_matrix.array().max(r_matrix.array());
				}
			}
		}
	}

};

template<typename Distance>
class RankedDistance {
private:
	Distance m_distance;
	ArrayXf m_percentiles;

public:
	inline RankedDistance(const Distance p_distance = Distance()) : m_distance(p_distance) {
	}

	inline float operator()(
		const WordVectors &p_vectors,
		token_t p_s,
		token_t p_t) const {

		const float d = m_distance(p_vectors, p_s, p_t);

		const float *begin = &m_percentiles[0];
		const float *end = &m_percentiles[m_percentiles.rows()];
		const float *mark = std::lower_bound(begin, end, d);

		if (mark <= begin) {
			return 0.0f;
		} else if (mark >= end) {
			return 1.0f;
		} else {
			float d0 = mark[-1];
			float d1 = mark[0];

			constexpr float step = 1.0f / 1000.0f; // percentile
			float r = step * ((mark - begin - 1) + ((d - d0) / (d1 - d0)));

			return r;
		}
	}

	ArrayXf &percentiles() {
		return m_percentiles;
	}
};

/*template<typename Distance>
class RankedSimilarityMeasure : public SimilarityMeasure<RankedDistance<Distance>> {
private:

public:
	RankedSimilarityMeasure(
		const WordVectors &p_vectors,
		const Distance p_distance = Distance()) :

		SimilarityMeasure<RankedDistance<Distance>>(p_vectors, RankedDistance(p_distance)) {
	}

	virtual void load_percentiles(const std::string &p_path, const std::string &p_name) {
		const auto table = load_parquet_table(p_path + ".percentiles." + p_name + ".parquet");
		this->m_distance.percentiles().resize(table->num_rows(), 1);

		for_each_column<arrow::FloatType, float>(table, [this] (int i, auto v) {
			 this->m_distance.percentiles().col(i) = v;
		});
	}
};*/

struct CosineSimilarity {
	inline float operator()(
		const WordVectors &p_vectors,
		token_t p_s,
		token_t p_t) const {

		return p_vectors.normalized.row(p_s).dot(p_vectors.normalized.row(p_t));
	}
};

struct SqrtCosine {
	inline float operator()(
		const WordVectors &p_vectors,
		token_t p_s,
		token_t p_t) const {

		float denom = p_vectors.raw.row(p_s).sum() * p_vectors.raw.row(p_t).sum();
		return (p_vectors.raw.row(p_s) * p_vectors.raw.row(p_t)).array().sqrt().sum() / denom;
	}
};

struct PNormSimilarity {
	float m_p;
	float m_distance_scale;

	PNormSimilarity(float p = 2.0f, float scale = 1.0f) : m_p(p), m_distance_scale(scale) {
	}

	inline float operator()(
		const WordVectors &p_vectors,
		token_t p_s,
		token_t p_t) const {

		auto uv = p_vectors.normalized.row(p_s) - p_vectors.normalized.row(p_t);
		float distance = pow(uv.cwiseAbs().array().pow(m_p).sum(), 1.0f / m_p);
		return std::max(0.0f, 1.0f - distance * m_distance_scale);
	}
};

struct NICDMSimilarity {
	// see Schnitzer, Dominik, et al.: Local and Global Scaling Reduce Hubs in Space.

	inline float operator()(
		const WordVectors &p_vectors,
		token_t p_s,
		token_t p_t) const {

		const int which = p_vectors.neighborhood.cols() - 1; // maximum context size

		const float mu_s = p_vectors.neighborhood(p_s, which);
		const float mu_t = p_vectors.neighborhood(p_t, which);

		const float scale = std::sqrt(mu_s * mu_t);

		float d_st = (p_vectors.normalized.row(p_s) - p_vectors.normalized.row(p_t)).norm();
		float nicdm = d_st / scale;

		// note: for scale == 1 this gives identical results to cosine similarity, as it is
		// then just a squared euclidean distance.

		return 1.0f - (nicdm * nicdm) / 2.0f;
	}
};

struct APSynPSimilarity {

	float m_min_score;
	float m_max_score;

	APSynPSimilarity(const int n_dim, const float apsynp_p) {
		float max_score = 0.0f;
		for (int i = 1; i <= n_dim; i++) {
			max_score += 1.0f / std::pow(i, apsynp_p);
		}
		m_max_score = max_score;

		m_min_score = n_dim * (1.0f / std::pow((1 + n_dim) / 2.0f, apsynp_p));
	}

	inline float operator()(
		const WordVectors &p_vectors,
		token_t p_s,
		token_t p_t) const {

		float s = ((p_vectors.apsynp.row(p_s) + p_vectors.apsynp.row(p_t)) / 2.0f).array().inverse().sum();

		s = (s - m_min_score) / (m_max_score - m_min_score);

		return s;
	}
};

std::map<std::string, EmbeddingSimilarityRef> create_similarity_measures(
	const std::string &p_name,
	const WordVectors &p_vectors) {

	// cosine deterioriates in high dimensions, see:
	// Radovanović, Milos, et al. “On the Existence of Obstinate Results in Vector Space Models.” Proceeding of the
	// 33rd International ACM SIGIR Conference on Research and Development in Information Retrieval - SIGIR ’10,
	// ACM Press, 2010, p. 186. DOI.org (Crossref), doi:10.1145/1835449.1835482.

	std::map<std::string, EmbeddingSimilarityRef> measures;

	measures["cosine"] = std::make_shared<SimilarityMeasure<CosineSimilarity>>(p_vectors);

	//measures["ranked-cosine"] = std::make_shared<RankedSimilarityMeasure<CosineSimilarity>>(p_vectors);

	measures["sqrt-cosine"] = std::make_shared<SimilarityMeasure<SqrtCosine>>(p_vectors);

	/*measures["maximum"] = std::make_shared<MaximumSimilarityMeasure>(std::vector<EmbeddingSimilarityRef>{
		measures["cosine"], measures["nicdm"], measures["apsynp"]});

	measures["ranked-maximum"] = std::make_shared<MaximumSimilarityMeasure>(std::vector<EmbeddingSimilarityRef>{
		measures["ranked-cosine"], measures["ranked-nicdm"], measures["ranked-apsynp"]});*/

#if 0
	// low pnorms can make sense, see
	// Aggarwal, Charu C., et al. “On the Surprising Behavior of Distance Metrics in High Dimensional Space.”
	// Database Theory — ICDT 2001, edited by Jan Van den Bussche and Victor Vianu, vol. 1973,
	// Springer Berlin Heidelberg, 2001, pp. 420–34. DOI.org (Crossref), doi:10.1007/3-540-44503-X_27.

	measures["pnorm-2"] = std::make_shared<PNormEmbeddingSimilarity>(p_vectors, PNormSimilarity(2.0f, 1.0f / 2.0f));
	measures["pnorm-1"] = std::make_shared<PNormEmbeddingSimilarity>(p_vectors, PNormSimilarity(1.0f, 1.0f));
	measures["pnorm-0.5"] = std::make_shared<PNormEmbeddingSimilarity>(p_vectors, PNormSimilarity(0.5f, 1.0f / 5000.0f));
	measures["pnorm-0.1"] = std::make_shared<PNormEmbeddingSimilarity>(p_vectors, PNormSimilarity(0.1f, 1.0f / 1000000000.0f));
#endif

	return measures;
}

template<typename T>
class String2Int {
	std::unordered_map<std::string, T> m_strings;

public:
	inline T operator[](const std::string &s) {
		const auto r = m_strings.insert({s, m_strings.size()});
		return r.first->second;
	}

	inline T lookup(const std::string &s) const {
		 const auto r = m_strings.find(s);
		 return r == m_strings.end() ? -1 : r->second;
	}

	std::string inverse_lookup_slow(T t) const {
		for (auto x : m_strings) {
			if (x.second == t) {
				return x.first;
			}
		}
		return "";
	}

	inline size_t size() const {
		return m_strings.size();
	}
};

template<typename T>
class StringLexicon {
	std::unordered_map<std::string, T> m_to_int;
	std::vector<std::string> m_to_str;

public:
	inline T add(const std::string &p_s) {
		const auto r = m_to_int.insert({p_s, m_to_str.size()});
		if (r.second) {
			m_to_str.push_back(p_s);
		}
		return r.first->second;
	}

	inline T lookup(const std::string &p_s) const {
		const auto i = m_to_int.find(p_s);
		if (i != m_to_int.end()) {
			return i->second;
		} else {
			return -1;
		}
	}

	inline const std::string &lookup(T p_id) const {
		return m_to_str.at(p_id);
	}

	inline size_t size() const {
		return m_to_int.size();
	}
};

class Vocabulary {
	String2Int<token_t> m_tokens;

	std::vector<float> m_idf;
	// we use this as a form of inverse document frequency
	// which is independent of our specific corpus documents.

	struct Embedding {
		EmbeddingRef embedding;
		std::vector<token_t> map;
	};

	std::vector<Embedding> m_embeddings;
	StringLexicon<int8_t> m_pos;
	StringLexicon<int8_t> m_tag;

	int m_det_pos;

public:
	// basically a mapping from token -> int

	std::mutex m_mutex;

	Vocabulary() : m_det_pos(-1) {
	}

	int add_embedding(EmbeddingRef p_embedding) {
		std::lock_guard<std::mutex> lock(m_mutex);
		PPK_ASSERT(m_tokens.size() == 0);
		Embedding e;
		e.embedding = p_embedding;
		m_embeddings.push_back(e);
		return m_embeddings.size() - 1;
	}

	inline int unsafe_add_pos(const std::string &p_name) {
		const int i = m_pos.add(p_name);
	    if (p_name == "DET") {
	        m_det_pos = i;
	    }
	    return i;
	}

	inline int det_pos() const {
	    return m_det_pos;
	}

	inline int unsafe_add_tag(const std::string &p_name) {
		return m_tag.add(p_name);
	}

	inline token_t unsafe_lookup(const std::string &p_token) const {
		return m_tokens.lookup(p_token);
	}

	inline token_t unsafe_add(const std::string &p_token) {
		const size_t old_size = m_tokens.size();
		const token_t t = m_tokens[p_token];
		if (m_tokens.size() > old_size) { // new token?
			for (Embedding &e : m_embeddings) {
				e.map.push_back(e.embedding->lookup(p_token));
			}
			m_idf.push_back(0.0f);
		}
		return t;
	}

	void set_idf_from_prob(token_t p_token, float p_prob) {
		// spaCy uses log (nt / N), but we want log (N / nt)
		// log N - log nt = -(log nt - log N) = -log (nt / N)

		m_idf.at(p_token) = -p_prob;
	}

	inline std::string token_to_string_slow(token_t p_token) {
		return m_tokens.inverse_lookup_slow(p_token);
	}

	inline const std::string &pos_str(int8_t p_pos_id) {
		return m_pos.lookup(p_pos_id);
	}

	inline const std::string &tag_str(int8_t p_tag_id) {
		return m_tag.lookup(p_tag_id);
	}

	POSWMap mapped_pos_weights(const std::map<std::string, float> &p_pos_weights) const {
		POSWMap pos_weights;
		for (auto const &x : p_pos_weights) {
			const int i = m_tag.lookup(x.first);
			if (i >= 0) {
				pos_weights[i] = x.second;
			}
		}
		return pos_weights;
	}

	std::map<std::string, MetricRef> create_metrics(
		const std::string &p_needle_text,
		const std::vector<Token> &p_needle,
		const std::set<std::string> p_needed_metrics,
		const std::string &p_embedding_similarity,
		float p_pos_mismatch_penalty,
		float p_similarity_falloff,
		float p_similarity_threshold,
		const POSWMap &p_pos_weights,
		const float p_idf_weight) {

		std::lock_guard<std::mutex> lock(m_mutex);

		std::map<std::string, MetricRef> metrics;

		for (const Embedding &embedding : m_embeddings) {
			if (p_needed_metrics.find(embedding.embedding->name()) == p_needed_metrics.end()) {
				continue;
			}

			const Eigen::Map<Eigen::Array<token_t, Eigen::Dynamic, 1>> vocabulary_ids(
				const_cast<token_t*>(embedding.map.data()), embedding.map.size());

			auto metric = embedding.embedding->create_metric(
				vocabulary_ids,
				m_idf,
				p_embedding_similarity,
				p_needle_text,
				p_needle,
				p_pos_mismatch_penalty,
				p_similarity_falloff,
				p_similarity_threshold,
				p_pos_weights,
				p_idf_weight);

			metrics[metric->name()] = metric;
		}

		return metrics;
	}
};

typedef std::shared_ptr<Vocabulary> VocabularyRef;



void init_pyarrow() {
	if (arrow::py::import_pyarrow() != 0) {
		std::cerr << "error initializing pyarrow.\n";
	}
}

template<typename T, typename TV>
const std::vector<TV> numeric_column(
	const std::shared_ptr<arrow::Table> &table,
	const std::string &field) {

	const int i = table->schema()->GetFieldIndex(field);
	if (i < 0) {
		throw std::runtime_error("extract_raw_values: illegal field name");
	}
	auto data = column_data(table->column(i));

	std::vector<TV> values;
    int64_t count = 0;
    for (int64_t k = 0; k < data->num_chunks(); k++) {
    	auto array = data->chunk(k);
        count += array->length();
    }

    values.resize(count);
    int64_t write_offset = 0;

    for (int64_t k = 0; k < data->num_chunks(); k++) {
    	auto array = data->chunk(k);

        if (array->type_id() != arrow::TypeTraits<T>::type_singleton()->id()) {
            std::stringstream s;
            s << "extract_raw_values: wrong data type " <<
                array->type()->name() << " != " << arrow::TypeTraits<T>::type_singleton()->name();
            throw std::runtime_error(s.str());
        }

    	auto num_array = std::static_pointer_cast<arrow::NumericArray<T>>(array);
    	const auto *raw = num_array->raw_values();
    	std::copy(raw, raw + array->length(), &values[write_offset]);
    	write_offset += array->length();
    }

    return values;
}

/*template<typename T>
const typename arrow::NumericArray<T>::value_type *extract_dict_indices(
	const std::shared_ptr<arrow::Table> &table,
	const std::string &field,
	int64_t &r_length) {

	const int i = table->schema()->GetFieldIndex(field);
	if (i < 0) {
		throw std::runtime_error("extract_raw_values: illegal field name");
	}

	auto data = table->column(i)->data();
	if (data->num_chunks() != 1) {
		throw std::runtime_error("extract_raw_values: chunks != 1");
	}
	auto array = data->chunk(0);


	auto dict_array = std::dynamic_pointer_cast<arrow::DictionaryArray>(array);
	if (!dict_array) {
		throw std::runtime_error("not a dictionary");
	}
	auto indices = dict_array->indices();

	if (indices->type_id() != arrow::TypeTraits<T>::type_singleton()->id()) {
		std::stringstream s;
		s << "extract_raw_values: wrong data type " <<
			indices->type()->name() << " != " << arrow::TypeTraits<T>::type_singleton()->name();
		throw std::runtime_error(s.str());
	}

	auto num_indices = std::static_pointer_cast<arrow::NumericArray<T>>(indices);
	r_length = num_indices->length();
	return num_indices->raw_values();
}*/

template<typename F>
class StringVisitor : public arrow::ArrayVisitor {
	F f;
	size_t m_index;

public:
	StringVisitor(const F &f) : f(f), m_index(0) {
	}

	virtual arrow::Status Visit(const arrow::StringArray &array) {
		const int n = array.length();
		for (int i = 0; i < n; i++) {
			f(m_index++, array.GetString(i));
		}
		return arrow::Status::OK();
	}
};

/*std::shared_ptr<arrow::StringArray> string_column(
    const std::shared_ptr<arrow::Table> &table,
    const std::string &field) {

    const int i = table->schema()->GetFieldIndex(field);
    if (i < 0) {
        throw std::runtime_error("extract_raw_values: illegal field name");
    }
    auto data = column_data(table->column(i));

    for (size_t k = 0; k < data->num_chunks(); k++) {
        auto array = data->chunk(k);
        PPK_ASSERT(array->type_id() == arrow::TypeTraits<arrow::StringType>::type_singleton()->id());
        auto s = std::dynamic_pointer_cast<arrow::StringArray>(array);
        PPK_ASSERT(s);
    }

}*/

template<typename F>
void iterate_strings(
	const std::shared_ptr<arrow::Table> &table,
	const std::string &field,
	const F &f) {

	const int i = table->schema()->GetFieldIndex(field);
	if (i < 0) {
		throw std::runtime_error("extract_raw_values: illegal field name");
	}
	auto data = column_data(table->column(i));
    StringVisitor v(f);
    for (int64_t k = 0; k < data->num_chunks(); k++) {
        auto array = data->chunk(k);
        ensure_type<arrow::StringType>(array);

        if (!array->Accept(&v).ok()) {
            throw std::runtime_error("arrow iteration error in iterate_strings");
        }
    }
}

template<typename F>
class FloatVisitor : public arrow::ArrayVisitor {
	F f;
	size_t m_index;

public:
	FloatVisitor(const F &f) : f(f), m_index(0) {
	}

	virtual arrow::Status Visit(const arrow::FloatArray &array) {
		const int n = array.length();
		for (int i = 0; i < n; i++) {
			f(m_index++, array.Value(i));
		}
		return arrow::Status::OK();
	}

	virtual arrow::Status Visit(const arrow::DoubleArray &array) {
		const int n = array.length();
		for (int i = 0; i < n; i++) {
			f(m_index++, array.Value(i));
		}
		return arrow::Status::OK();
	}
};

template<typename F>
void iterate_floats(
	const std::shared_ptr<arrow::Table> &table,
	const std::string &field,
	const F &f) {

	const int i = table->schema()->GetFieldIndex(field);
	if (i < 0) {
		throw std::runtime_error("extract_raw_values: illegal field name");
	}
	auto data = column_data(table->column(i));
	FloatVisitor v(f);
    for (int64_t k = 0; k < data->num_chunks(); k++) {
        auto array = data->chunk(k);
        ensure_type<arrow::DoubleType>(array);
        if (!array->Accept(&v).ok()) {
            throw std::runtime_error("arrow iteration error in iterate_floats");
        }
    }
}

class FastEmbedding : public Embedding {
	std::unordered_map<std::string, long> m_tokens;
	WordVectors m_embeddings;
	std::map<std::string, EmbeddingSimilarityRef> m_similarity_measures;

public:
	FastEmbedding(
		const std::string &p_name,
		py::object p_table) : Embedding(p_name) {

		const std::shared_ptr<arrow::Table> table = unwrap_table(p_table);

		iterate_strings(table, "token", [this] (size_t i, const std::string &s) {
			m_tokens[s] = static_cast<long>(i);
		});

		std::cout << p_name << ": " << "loaded " << m_tokens.size() << " tokens." << std::endl;

		/*{
			auto tokens = string_column(table, "token");

			if (!tokens) {
				throw std::runtime_error("failed to find parquet table column 'token'\n");
			}

			n = tokens->length();
			m_tokens.reserve(n);
			for (int i = 0; i < n; i++) {
				m_tokens[tokens->GetString(i)] = i;
			}
		}*/
		//const size_t n = m_tokens.length();

		// note: these "raw" tables were already normalized in preprocessing.

		m_embeddings.raw.resize(m_tokens.size(), table->num_columns() - 1);

		try {
			/*printf("loading embedding vectors parquet table.\n");
			fflush(stdout);*/

			for_each_column<arrow::FloatType, float>(table, [this] (int i, auto v, auto offset) {
				m_embeddings.raw.col(i - 1)(Eigen::seqN(offset, v.size())) = v.max(0);
			}, 1);
		} catch(...) {
			printf("failed to load embedding vectors parquet table.\n");
			throw;
		}

		m_embeddings.update_normalized();

		m_similarity_measures = create_similarity_measures(p_name, m_embeddings);
	}

	void add_apsynp(py::object p_table, float apsynp_p = 0.1) {
		const std::shared_ptr<arrow::Table> table = unwrap_table(p_table);

		if (size_t(table->num_rows()) != m_tokens.size()) {
			fprintf(stderr, "apsynp table: token size %ld != %ld\n",
				long(table->num_rows()), long(m_tokens.size()));
			throw std::runtime_error("broken table");
		}
		m_embeddings.apsynp.resize(m_tokens.size(), table->num_columns());

		try {
			for_each_column<arrow::UInt16Type, uint16_t>(table, [this, apsynp_p] (int i, auto v, auto offset) {
				m_embeddings.apsynp.col(i)(Eigen::seqN(offset, v.size())) = v.template cast<float>().pow(apsynp_p);
			});
		} catch(...) {
			printf("failed to load apsynp parquet table.");
			throw;
		}

		m_similarity_measures["apsynp"] = std::make_shared<SimilarityMeasure<APSynPSimilarity>>(
			m_embeddings, APSynPSimilarity(m_embeddings.raw.cols(), apsynp_p));

		//m_similarity_measures["ranked-apsynp"] = std::make_shared<RankedSimilarityMeasure<APSynPSimilarity>>(
		//	m_embeddings, APSynPSimilarity(m_embeddings.raw.cols(), apsynp_p));
	}

	void add_nicdm(py::object p_table) {
		const std::shared_ptr<arrow::Table> table = unwrap_table(p_table);

		if (size_t(table->num_rows()) != m_tokens.size()) {
			fprintf(stderr, "nicdm table: token size %ld != %ld\n",
				long(table->num_rows()), long(m_tokens.size()));
			throw std::runtime_error("broken table");
		}
		m_embeddings.neighborhood.resize(m_tokens.size(), table->num_columns());

		try {
			for_each_column<arrow::FloatType, float>(table, [this] (int i, auto v, auto offset) {
				m_embeddings.neighborhood.col(i)(Eigen::seqN(offset, v.size())) = v;
			});
		} catch(...) {
			printf("failed to load neighborhood parquet table.\n");
			throw;
		}

		m_similarity_measures["nicdm"] = std::make_shared<SimilarityMeasure<NICDMSimilarity>>(m_embeddings);

		//m_similarity_measures["ranked-nicdm"] = std::make_shared<RankedSimilarityMeasure<NICDMSimilarity>>(m_embeddings);
	}

	void load_percentiles(const std::string &p_path) {
		for (auto i : m_similarity_measures) {
			i.second->load_percentiles(p_path, i.first);
		}
	}

	virtual MetricRef create_metric(
		const TokenIdArray &p_vocabulary_to_embedding,
		const std::vector<float> &p_idf,
		const std::string &p_embedding_similarity,
		const std::string &p_needle_text,
		const std::vector<Token> &p_needle,
		float p_pos_mismatch_penalty,
		float p_similarity_falloff,
		float p_similarity_threshold,
		const POSWMap &p_pos_weights,
		const float p_idf_weight) {

		auto m = std::make_shared<FastMetric>(
			shared_from_this(),
			p_pos_mismatch_penalty,
			p_similarity_falloff,
			p_similarity_threshold,
			p_pos_weights);

		auto s = m_similarity_measures.find(p_embedding_similarity);
		if (s == m_similarity_measures.end()) {
			throw std::runtime_error(
				std::string("unknown similary measure ") + p_embedding_similarity);
		}

		WeightedIDF vocab_prob;
		vocab_prob.idf = &p_idf;
		vocab_prob.w = p_idf_weight;

		build_similarity_matrix(
			p_vocabulary_to_embedding,
			p_needle_text,
			p_needle,
			s->second,
			&vocab_prob,
			m->w_similarity());

		m->w_similarity() = m->w_similarity().array().pow(p_similarity_falloff);

		return m;
	}

	float cosine_similarity(const std::string &p_a, const std::string &p_b) const {
		const auto a = m_tokens.find(p_a);
		const auto b = m_tokens.find(p_b);
		if (a != m_tokens.end() && b != m_tokens.end()) {
			return m_embeddings.raw.row(a->second).dot(m_embeddings.raw.row(b->second));
		} else {
			return 0.0f;
		}
	}

	MatrixXf similarity_matrix(
		const std::string &p_measure,
		TokenIdArray p_s_embedding_ids,
		TokenIdArray p_t_embedding_ids) const {

		auto i = m_similarity_measures.find(p_measure);
		if (i == m_similarity_measures.end()) {
			throw std::runtime_error("unknown similarity measure");
		}

		MatrixXf m;

		i->second->build_matrix(
			m_embeddings, p_s_embedding_ids, p_t_embedding_ids, nullptr, m);

		return m;
	}

	token_t lookup(const std::string &p_token) const {
		const auto i = m_tokens.find(p_token);
		if (i != m_tokens.end()) {
			return i->second;
		} else {
			return -1;
		}
	}

	size_t n_tokens() const {
		return m_embeddings.raw.rows();
	}

	py::list measures() const {
		py::list names;
		for (auto i : m_similarity_measures) {
			names.append(py::str(i.first));
		}
		return names;
	}

private:
	void build_similarity_matrix(
		const TokenIdArray &p_vocabulary_to_embedding,
		const std::string &p_needle_text,
		const std::vector<Token> &p_needle,
		const EmbeddingSimilarityRef &p_embedding_similarity,
		const WeightedIDF *p_prob,
		MatrixXf &r_matrix) const {

		TokenIdArray needle_vocabulary_token_ids;
		needle_vocabulary_token_ids.resize(p_needle.size());

		for (size_t i = 0; i < p_needle.size(); i++) {
			needle_vocabulary_token_ids[i] = p_needle[i].id;
		}

		// p_a maps from a Vocabulary corpus token id to an Embedding token id,
		// e.g. 3 in the corpus and 127 in the embedding.

		// p_b are the needle's Vocabulary token ids (not yet mapped to Embedding)

		TokenIdArray needle_embedding_token_ids;
		needle_embedding_token_ids.resize(p_needle.size());

		for (size_t i = 0; i < p_needle.size(); i++) {
			const token_t t = needle_vocabulary_token_ids[i];
			if (t >= 0) {
				PPK_ASSERT(t < p_vocabulary_to_embedding.rows());
				needle_embedding_token_ids[i] =
					p_vocabulary_to_embedding[t]; // map to Embedding token ids
			} else {
				// that word is not in our Vocabulary, it might be in the Embedding though.
				const auto word = p_needle_text.substr(p_needle.at(i).idx, p_needle.at(i).len);
				std::string lower = py::str(py::str(word).attr("lower")());
				needle_embedding_token_ids[i] = lookup(lower);
			}
			//std::cout << "xx mapped " << t << " -> " << p_a[t] << std::endl;
		}

		py::gil_scoped_release release;

		p_embedding_similarity->build_matrix(
			m_embeddings,
			p_vocabulary_to_embedding,
			needle_embedding_token_ids,
			p_prob,
			r_matrix);

		for (size_t j = 0; j < p_needle.size(); j++) { // for each token in needle

			// since the j-th needle token is a specific vocabulary token, we always
			// set that specific vocabulary token similarity to 1 (regardless of the
			// embedding distance).
			if (needle_vocabulary_token_ids[j] >= 0) {
				r_matrix(needle_vocabulary_token_ids[j], j) = 1.0f;
			}
		}

	}
};

typedef std::shared_ptr<FastEmbedding> FastEmbeddingRef;

std::vector<Sentence> unpack_sentences(const std::shared_ptr<arrow::Table> &p_table) {
	py::gil_scoped_release release;

	const auto book = numeric_column<arrow::Int8Type, int8_t>(p_table, "book");
	const auto chapter = numeric_column<arrow::Int8Type, int8_t>(p_table, "chapter");
	const auto speaker = numeric_column<arrow::Int8Type, int8_t>(p_table, "speaker");
	const auto location = numeric_column<arrow::UInt16Type, uint16_t>(p_table, "location");

	const auto n_tokens_values = numeric_column<arrow::UInt16Type, uint16_t>(p_table, "n_tokens");

	const size_t n = n_tokens_values.size();
	std::vector<Sentence> sentences;
	sentences.reserve(n);

	int32_t token_at = 0;
	for (size_t i = 0; i < n; i++) {
		 Sentence s;
		 s.book = book[i];
		 s.chapter = chapter[i];
		 s.speaker = speaker[i];
		 s.paragraph = location[i];
		 s.n_tokens = n_tokens_values[i];

		 s.token_at = token_at;
		 token_at += s.n_tokens;

		 sentences.push_back(s);
	}

	return sentences;
}

enum VocabularyMode {
	MODIFY_VOCABULARY,
	DO_NOT_MODIFY_VOCABULARY
};

typedef std::shared_ptr<std::vector<Token>> TokenVectorRef;

TokenVectorRef unpack_tokens(
	VocabularyRef p_vocab,
	VocabularyMode p_mode,
	const std::string p_text,
	const std::shared_ptr<arrow::Table> &p_table) {

	const auto idx = numeric_column<arrow::UInt32Type, uint32_t>(p_table, "idx");
	const auto len = numeric_column<arrow::UInt8Type, uint8_t>(p_table, "len");

	/*const auto pos_array = string_column(p_table, "pos");
	const auto tag_array = string_column(p_table, "tag");*/

	const size_t n = idx.size();
	PPK_ASSERT(n == len.size());

	TokenVectorRef tokens_ref = std::make_shared<std::vector<Token>>();

	std::vector<Token> &tokens = *tokens_ref.get();
	std::vector<std::string> token_texts;

	{
    	py::gil_scoped_release release;

    	tokens.reserve(n);
    	token_texts.reserve(n);

        for (size_t i = 0; i < n; i++) {
            if (idx[i] + len[i] > p_text.length()) {
                std::ostringstream s;
                s << "illegal token idx @" << i << "/" << n << ": " <<
                    idx[i] << " + " << len[i] <<
                    " > " << p_text.length();
                throw std::runtime_error(s.str());
            }

            token_texts.push_back(p_text.substr(idx[i], len[i]));
        }
    }

    py::object sub = (py::object) py::module::import("re").attr("sub");

    auto pattern = py::str("[^\\w]");
    auto repl = py::str("");

    for (size_t i = 0; i < n; i++) {
        auto &t = token_texts[i];
        auto s = py::str(py::str(t).attr("lower")());
		t = py::str(sub(pattern, repl, s));
    }

    {
    	py::gil_scoped_release release;

    	std::lock_guard<std::mutex> lock(p_vocab->m_mutex);

        for (size_t i = 0; i < n; i++) {
            Token t;
            const std::string &token_text = token_texts[i];

            // std::cout << "token: " << token_text << " " << int(len[i]) << std::endl;
            t.id = (p_mode == MODIFY_VOCABULARY) ?
                p_vocab->unsafe_add(token_text) :
                p_vocab->unsafe_lookup(token_text);

            t.idx = idx[i];
            t.len = len[i];
            t.pos = -1;
            t.tag = -1;
            tokens.push_back(t);
        }

        iterate_strings(p_table, "pos", [&tokens, p_vocab] (size_t i, const std::string &s) {
            tokens.at(i).pos = p_vocab->unsafe_add_pos(s);
        });

        iterate_strings(p_table, "tag", [&tokens, p_vocab] (size_t i, const std::string &s) {
            tokens.at(i).tag = p_vocab->unsafe_add_tag(s);
        });

        if (p_mode == MODIFY_VOCABULARY) {
            iterate_floats(p_table, "prob", [&tokens, p_vocab] (size_t i, float p) {
                p_vocab->set_idf_from_prob(tokens.at(i).id, p);
            });
        }
	}

	return tokens_ref;
}

class MismatchPenalty {
	std::vector<float> m_penalties;

public:
	MismatchPenalty(float cutoff, int max_n) {
		m_penalties.resize(max_n);

		if (cutoff < 0.0f) { // disabled / off?
			for (int i = 0; i < max_n; i++) {
				m_penalties[i] = 0.0f;
			}
		} else {
			// this cumbersome formulation is equivalent to (1 - 2^-(i / cutoff))

			const float scale = cutoff / 0.693147;

			for (int i = 0; i < max_n; i++) {
				m_penalties[i] = std::min(1.0f, 1.0f - exp(-i / scale));
			}
		}
	}

	inline double operator()(const size_t x) const {
		if (x < m_penalties.size()) {
			return m_penalties.at(x);
		} else {
			return 1e10;
		}
	}
};

typedef std::shared_ptr<MismatchPenalty> MismatchPenaltyRef;

class MismatchLength {
	int16_t m_length;
	float m_penalty;

public:
	inline MismatchLength() {
	}

	inline MismatchLength(const MismatchLength &l) :
		m_length(l.m_length),
		m_penalty(l.m_penalty)
	{
	}

	inline static MismatchLength empty() {
		MismatchLength l;
		l.m_length = 0;
		l.m_penalty = 0.0f;
		return l;
	}

	inline MismatchLength extended_by(const MismatchPenalty &p, int n) const {
		MismatchLength l;
		l.m_length = m_length + n;
		l.m_penalty = p(l.m_length);
		return l;
	}

	inline int len() const {
		return m_length;
	}

	inline float penalized(const float p_score) const {
		return p_score * std::max(0.0f, 1.0f - m_penalty);
	}
};

class Region {
	const std::string m_s;
	const float m_mismatch_penalty;

public:
	Region(
		std::string &&s,
		float p_mismatch_penalty = 0.0f):

		m_s(s),
		m_mismatch_penalty(p_mismatch_penalty) {
	}

	virtual ~Region() {
	}

	py::bytes s() const {
		return m_s;
	}

	float mismatch_penalty() const {
		return m_mismatch_penalty;
	}

	virtual bool is_matched() const {
		return false;
	}
};

typedef std::shared_ptr<Region> RegionRef;

struct TokenRef {
	TokenVectorRef tokens;
	int32_t index;

	const Token *operator->() const {
		return &tokens->at(index);
	}
};

struct TokenScore {
	float similarity;
	float weight;
};

class MatchedRegion : public Region {
	const TokenScore m_score; // score between s and t
	const std::string m_t;

	const VocabularyRef m_vocab;
	const TokenRef m_s_token;
	const TokenRef m_t_token;

	const std::string m_metric;

public:
	MatchedRegion(
		const TokenScore &p_score,
		std::string &&s,
		std::string &&t,
		const VocabularyRef &p_vocab,
		const TokenRef &p_s_token,
		const TokenRef &p_t_token,
		const std::string &p_metric) :

		Region(std::move(s)),
		m_score(p_score),
		m_t(t),

		m_vocab(p_vocab),
		m_s_token(p_s_token),
		m_t_token(p_t_token),

		m_metric(p_metric) {
		}

		virtual bool is_matched() const {
			return true;
		}

		float similarity() const {
			return m_score.similarity;
		}

		float weight() const {
			return m_score.weight;
		}

		py::bytes t() const {
			return m_t;
		}

		py::bytes pos_s() const {
			return m_vocab->pos_str(m_s_token->pos);
		}

		py::bytes pos_t() const {
			return m_vocab->pos_str(m_t_token->pos);
		}

		py::bytes metric() const {
			return m_metric;
		}
};

typedef std::shared_ptr<MatchedRegion> MatchedRegionRef;

static MetricRef lookup_metric(
	const std::map<std::string, MetricRef> &p_metrics,
	const std::string &p_name) {

	const auto i = p_metrics.find(p_name);
	if (i == p_metrics.end()) {
		throw std::runtime_error(
			std::string("could not find a metric named ") + p_name);
	}
	return i->second;
}

class Query : public std::enable_shared_from_this<Query> {
	std::vector<MetricRef> m_metrics;
	const std::string m_text;
	TokenVectorRef m_t_tokens;
	POSWMap m_pos_weights;
	std::vector<float> m_t_tokens_pos_weights;
	float m_total_score;
	std::string m_cost_combine_function;
	int m_mismatch_length_penalty;
	float m_submatch_weight;
	bool m_bidirectional;
	bool m_ignore_determiners;
	bool m_aborted;

	/*void init_boost(const py::kwargs &p_kwargs) {
		m_t_boost.reserve(m_t_tokens.size());

		if (p_kwargs && p_kwargs.contains("t_boost")) {
			auto t_boost = p_kwargs["t_boost"].cast<py::list>();
			for (size_t i = 0; i < m_t_tokens.size(); i++) {
				m_t_boost.push_back(t_boost[i].cast<float>());
			}
		} else {
			for (size_t i = 0; i < m_t_tokens.size(); i++) {
				m_t_boost.push_back(1.0f);
			}
		}
	}*/

public:
	Query(
		VocabularyRef p_vocab,
		const std::string &p_text,
		py::handle p_tokens_table,
		py::kwargs p_kwargs) : m_text(p_text), m_aborted(false) {

		const std::shared_ptr<arrow::Table> table(
		    unwrap_table(p_tokens_table.ptr()));

		m_t_tokens = unpack_tokens(
			p_vocab, DO_NOT_MODIFY_VOCABULARY, p_text, table);

		//init_boost(p_kwargs);

		const float pos_mismatch_penalty =
			(p_kwargs && p_kwargs.contains("pos_mismatch_penalty")) ?
				p_kwargs["pos_mismatch_penalty"].cast<float>() :
				1.0f;

		const float similarity_threshold = (p_kwargs && p_kwargs.contains("similarity_threshold")) ?
            p_kwargs["similarity_threshold"].cast<float>() :
            0.0f;

		const float similarity_falloff = (p_kwargs && p_kwargs.contains("similarity_falloff")) ?
            p_kwargs["similarity_falloff"].cast<float>() :
            1.0f;

		m_submatch_weight = (p_kwargs && p_kwargs.contains("submatch_weight")) ?
            p_kwargs["submatch_weight"].cast<float>() :
            0.0f;

		m_bidirectional = (p_kwargs && p_kwargs.contains("bidirectional")) ?
            p_kwargs["bidirectional"].cast<bool>() :
            false;

		m_ignore_determiners = (p_kwargs && p_kwargs.contains("ignore_determiners")) ?
            p_kwargs["ignore_determiners"].cast<bool>() :
            false;

		const float idf_weight = (p_kwargs && p_kwargs.contains("idf_weight")) ?
				p_kwargs["idf_weight"].cast<float>() :
				0.0f;

		std::set<std::string> needed_metrics;
		if (p_kwargs && p_kwargs.contains("metrics")) {
			auto given_metrics = p_kwargs["metrics"].cast<py::list>();
			for (const auto &item : given_metrics) {
				if (py::isinstance<py::str>(item)) {
					const std::string name = item.cast<py::str>();
					needed_metrics.insert(name);
				} else if (py::isinstance<py::tuple>(item)) {
					auto tuple = item.cast<py::tuple>();
					if (tuple.size() != 3) {
						throw std::runtime_error("expected 3-tuple as metric");
					}
					const std::string a = tuple[0].cast<py::str>();
					const std::string b = tuple[1].cast<py::str>();
					needed_metrics.insert(a);
					needed_metrics.insert(b);
				} else {
						throw std::runtime_error(
							"expected list of 3-tuples as metrics");
				}
			}
		}

		std::map<std::string, float> pos_weights;
		if (p_kwargs && p_kwargs.contains("pos_weights")) {
			auto pws = p_kwargs["pos_weights"].cast<py::dict>();
			for (const auto &pw : pws) {
				pos_weights[pw.first.cast<py::str>()] = pw.second.cast<py::float_>();
			}
		}

		m_pos_weights = p_vocab->mapped_pos_weights(pos_weights);

		m_t_tokens_pos_weights.reserve(m_t_tokens->size());
		for (size_t i = 0; i < m_t_tokens->size(); i++) {
			const Token &t = m_t_tokens->at(i);

			auto w = m_pos_weights.find(t.tag);
			float s;
			if (w != m_pos_weights.end()) {
				s = w->second;
			} else {
				s = 1.0f;
			}

			m_t_tokens_pos_weights.push_back(s);
		}

		m_total_score = 0.0f;
		for (float w : m_t_tokens_pos_weights) {
			m_total_score += w;
		}

		const std::string similarity_measure = (p_kwargs && p_kwargs.contains("similarity_measure")) ?
			p_kwargs["similarity_measure"].cast<py::str>() : "cosine";

		auto metrics = p_vocab->create_metrics(
			m_text,
			*m_t_tokens.get(),
			needed_metrics,
			similarity_measure,
			pos_mismatch_penalty,
			similarity_falloff,
			similarity_threshold,
			m_pos_weights,
			idf_weight);

		// metrics are specified as list (m1, m2, ...) were each m is
		// either the name of a metric, e.g. "fasttext", or a 3-tuple
		// that specifies a mix: ("fasttext", "wn2vec", 0.2)

		if (p_kwargs && p_kwargs.contains("metrics")) {
			auto given_metrics = p_kwargs["metrics"].cast<py::list>();
			for (const auto &item : given_metrics) {
				if (py::isinstance<py::str>(item)) {
					const std::string name = item.cast<py::str>();
					m_metrics.push_back(lookup_metric(metrics, name));
				} else if (py::isinstance<py::tuple>(item)) {
					auto tuple = item.cast<py::tuple>();
					if (tuple.size() != 3) {
						throw std::runtime_error("expected 3-tuple as metric");
					}
					const std::string a = tuple[0].cast<py::str>();
					const std::string b = tuple[1].cast<py::str>();
					const float t = tuple[2].cast<float>();
					m_metrics.push_back(std::make_shared<CompositeMetric>(
						lookup_metric(metrics, a),
						lookup_metric(metrics, b),
						t
					));
				} else {
						throw std::runtime_error(
							"expected list as specification for metrics");
				}
			}
		}

		if (p_kwargs && p_kwargs.contains("cost_combine_function")) {
			m_cost_combine_function = p_kwargs["cost_combine_function"].cast<std::string>();
		} else {
			m_cost_combine_function = "sum";
		}

		if (p_kwargs && p_kwargs.contains("mismatch_length_penalty")) {
			auto penalty =  p_kwargs["mismatch_length_penalty"];
			if (py::isinstance<py::str>(penalty)) {
				const std::string x = penalty.cast<py::str>();
				if (x == "off") {
					m_mismatch_length_penalty = -1.0f; // off
				} else {
					throw std::runtime_error(
						"illegal value for mismatch_length_penalty");
				}
			} else {
				m_mismatch_length_penalty = penalty.cast<float>();
			}
		} else {
			m_mismatch_length_penalty = 5;
		}
	}

	const std::string &text() const {
		return m_text;
	}

	inline const TokenVectorRef &tokens() const {
		return m_t_tokens;
	}

	inline int len() const {
		return m_t_tokens->size();
	}

	inline const POSWMap &pos_weights() const {
		return m_pos_weights;
	}

	const std::vector<MetricRef> &metrics() const {
		return m_metrics;
	}

	const std::string &cost_combine_function() const {
		return m_cost_combine_function;
	}

	inline int mismatch_length_penalty() const {
		return m_mismatch_length_penalty;
	}

	inline bool bidirectional() const {
		return m_bidirectional;
	}

	inline bool ignore_determiners() const {
	    return m_ignore_determiners;
	}

	ResultSetRef match(
		const DocumentRef &p_document);

	bool aborted() const {
		return m_aborted;
	}

	void abort() {
		m_aborted = true;
	}

	inline int max_matches() const {
		return 100;
	}

	inline float min_score() const {
		return 0.2f;
	}

	inline float reference_score(
		const float p_matched,
		const float p_unmatched) const {

		// m_matched_weight == 0 indicates that there
		// is no higher relevance of matched content than
		// unmatched content, both are weighted equal (see
		// maximum_internal_score()).

		const float unmatched_weight = std::pow(
			(m_total_score - p_matched) / m_total_score,
			m_submatch_weight);

		const float reference_score =
			p_matched +
			unmatched_weight * (m_total_score - p_matched);

		return reference_score;
	}

	template<typename CostCombine>
	inline float normalized_score(
		const float p_raw_score,
		const std::vector<int16_t> &p_match) const {

#if 0
		return p_raw_score / m_total_score;

#else
		// FIXME: CostCombine is assumed to be sum right now.

		// a final boosting step allowing matched content
		// more weight than unmatched content.

		const size_t n = p_match.size();

		float matched_score = 0.0f;
		float unmatched_score = 0.0f;

		for (size_t i = 0; i < n; i++) {

			const float s = m_t_tokens_pos_weights[i];

			if (p_match[i] < 0) {
				unmatched_score += s;
			} else {
				matched_score += s;
			}
		}

		return p_raw_score / reference_score(matched_score, unmatched_score);
#endif
	}
};

typedef std::shared_ptr<Query> QueryRef;

class MatchDigest {
public:
	DocumentRef document;
	int32_t sentence_id;
	std::vector<int16_t> match;

	template<template<typename> typename C>
	struct compare;

	inline MatchDigest(
		DocumentRef p_document,
		int32_t p_sentence_id,
		const std::vector<int16_t> &p_match) :

		document(p_document),
		sentence_id(p_sentence_id),
		match(p_match) {
	}
};

class Match;

typedef std::shared_ptr<Match> MatchRef;

class Match {
private:
	const QueryRef m_query;
	const MetricRef m_metric;
	const int16_t m_scores_id;

	const MatchDigest m_digest;
	float m_score; // overall score
	std::vector<TokenScore> m_scores;

	int _pos_filter() const;

public:
	Match(
		const QueryRef &p_query,
		const MetricRef &p_metric,
		const int p_scores_id,
		MatchDigest &&p_digest,
		float p_score);

	inline float score() const {
		return m_score;
	}

	inline int32_t sentence_id() const {
		return m_digest.sentence_id;
	}

	inline const std::vector<int16_t> &match() const {
		return m_digest.match;
	}

	inline const Sentence &sentence() const;

	py::tuple location() const;

	py::list regions() const;

	py::list omitted() const;

	const std::string &metric() const {
		return m_metric->name();
	}

	inline const int scores_variant_id() const {
		return m_scores_id;
	}

	inline const DocumentRef &document() const {
		return m_digest.document;
	}

	template<template<typename> typename C>
	struct compare_by_score;

	using is_worse = compare_by_score<std::greater>;

	using is_better = compare_by_score<std::less>;

	template<typename Scores>
	void compute_scores(const Scores &p_scores, int p_len_s);

	void print_scores() const {
		for (auto s : m_scores) {
			printf("similarity: %f, weight: %f\n", s.similarity, s.weight);
		}
	}
};

typedef std::shared_ptr<Match> MatchRef;

class Document : public std::enable_shared_from_this<Document> {
private:
	const int64_t m_id;
	const VocabularyRef m_vocab;
	const std::string m_text;

	TokenVectorRef m_tokens;
	std::vector<Sentence> m_sentences;
	size_t m_max_len_s;

	const py::dict m_metadata;
	std::string m_cache_path;

public:
	Document(
		int64_t p_document_id,
		VocabularyRef p_vocab,
		const std::string &p_text,
		const py::object &p_sentences,
		const py::object &p_tokens,
		const py::dict &p_metadata,
		const std::string &p_cache_path);

	ResultSetRef find(const QueryRef &p_query);

	inline VocabularyRef vocabulary() const {
		return m_vocab;
	}

	std::string __str__() const {
		return "<cpp.vcore.Document " +
			m_metadata["author"].cast<std::string>() +
			", " +
			m_metadata["title"].cast<std::string>() + ">";
	}

	inline int64_t id() const {
		return m_id;
	}

	const std::string &path() const {
		return m_cache_path;
	}

	const std::string &text() const {
		return m_text;
	}

	const py::dict &metadata() const {
		return m_metadata;
	}

	inline const TokenVectorRef &tokens() const {
		return m_tokens;
	}

	size_t n_tokens() const {
		return m_tokens->size();
	}

	inline const std::vector<Sentence> &sentences() const {
		return m_sentences;
	}

	size_t n_sentences() const {
		return m_sentences.size();
	}

	inline size_t max_len_s() const { // maximum sentence length (in tokens)
		return m_max_len_s;
	}

	inline const Sentence &sentence(size_t p_index) const {
		return m_sentences.at(p_index);
	}

	py::list py_sentences_as_tokens() const {
		size_t k = 0;
		py::list py_doc;
		const auto &tokens = *m_tokens.get();
		for (const Sentence &s : m_sentences) {
			py::list py_sent;
			for (int i = 0; i < s.n_tokens; i++) {
				const auto &t = tokens[k++];
				py_sent.append(py::str(m_text.substr(t.idx, t.len)));
			}
			py_doc.append(py_sent);
		}
		return py_doc;
	}

	py::list py_sentences_as_text() const {
		py::list py_sentences;
		const auto &tokens = *m_tokens.get();
		for (const Sentence &s : m_sentences) {
			if (s.n_tokens > 0) {
				const auto &t0 = tokens[s.token_at];

				int32_t i1;
				if (s.token_at + s.n_tokens < static_cast<int32_t>(tokens.size())) {
					i1 = tokens[s.token_at + s.n_tokens].idx;
				} else {
					const auto &t1 = tokens[s.token_at + s.n_tokens - 1];
					i1 = t1.idx + t1.len;
				}

				py_sentences.append(py::str(m_text.substr(t0.idx, i1 - t0.idx)));
			} else {
				py_sentences.append(py::str(""));
			}
		}
		return py_sentences;
	}

};

typedef std::shared_ptr<Document> DocumentRef;

template<template<typename> typename C>
struct MatchDigest::compare {
	inline bool operator()(
		const MatchDigest &a,
		const MatchDigest &b) const {

		if (a.document == b.document) {
			if (C<int32_t>()(a.sentence_id, b.sentence_id)) {
				return true;
			} else {

				return std::lexicographical_compare(
					a.match.begin(), a.match.end(),
					b.match.begin(), b.match.end());

			}
		} else {
			PPK_ASSERT(a.document.get() && b.document.get());
			if (C<int64_t>()(a.document->id(), b.document->id())) {
				return true;
			}
		}

		return false;
	}
};

inline int Match::_pos_filter() const {
    return m_query->ignore_determiners() ?
        document()->vocabulary()->det_pos() : -1;
}

template<typename Scores>
void Match::compute_scores(const Scores &p_scores, int p_len_s) {
    const auto &match = m_digest.match;

    if (m_scores.empty() && !match.empty()) {
        const auto token_at = sentence().token_at;

        int end = 0;
        for (auto m : match) {
            end = std::max(end, int(m));
        }

        const auto sentence_scores = p_scores.create_sentence_scores(
            token_at, p_len_s, _pos_filter());
        m_scores.reserve(match.size());

        int i = 0;
        for (auto m : match) {
            if (m >= 0) {
                m_scores.push_back(TokenScore{
                    sentence_scores.similarity(m, i),
                    sentence_scores.weight(m, i)});
            } else {
                m_scores.push_back(TokenScore{0.0f, 0.0f});
            }
            i++;
        }
    }
}


template<template<typename> typename C>
struct Match::compare_by_score {
	inline bool operator()(
		const MatchRef &a,
		const MatchRef &b) const {

		if (C<float>()(a->score(), b->score())) {
			return true;
		} else if (a->score() == b->score()) {

			if (a->document() == b->document()) {

				if (C<int32_t>()(a->sentence_id(), b->sentence_id())) {
					return true;
				} else {

					return std::lexicographical_compare(
						a->match().begin(), a->match().end(),
						b->match().begin(), b->match().end());

				}
			} else {

				PPK_ASSERT(a->document().get() && b->document().get());

				if (C<int64_t>()(a->document()->id(), b->document()->id())) {
					return true;
				}
			}
		}
		return false;
	}
};

inline Match::Match(
	const QueryRef &p_query,
	const MetricRef &p_metric,
	const int p_scores_id,
	MatchDigest &&p_digest,
	float p_score) :

	m_query(p_query),
	m_metric(p_metric),
	m_scores_id(p_scores_id),
	m_digest(p_digest),
	m_score(p_score) {
}

inline const Sentence &Match::sentence() const {
	return  document()->sentence(sentence_id());
}

py::tuple Match::location() const {
	const auto &s = sentence();

	return py::make_tuple(
		s.book,
		s.chapter,
		s.speaker,
		s.paragraph
	);
}

py::list Match::regions() const {
	PPK_ASSERT(document().get() != nullptr);

	const std::string &s_text = document()->text();
	const std::string &t_text = m_query->text();
	const auto &s_tokens_ref = document()->tokens();
	const auto &t_tokens_ref = m_query->tokens();
	const std::vector<Token> &s_tokens = *s_tokens_ref.get();
	const std::vector<Token> &t_tokens = *t_tokens_ref.get();

	const auto token_at = sentence().token_at;

	const auto &match = this->match();
	const auto &scores = m_scores;

	PPK_ASSERT(match.size() > 0);
	PPK_ASSERT(match.size() == scores.size());

	int match_0 = 0;
	for (auto m : match) {
		if (m >= 0) {
			match_0 = m;
			break;
		}
	}

#if 0
	if (document()->id() == 17 && sentence_id() == 287) {
		printf("Match::regions\n");
		for (auto x : match) {
			printf("%d\n", x);
		}
	}
#endif

	constexpr int window_size = 10;
	int32_t last_anchor = std::max(0, token_at + match_0 - window_size);
	bool last_matched = false;

	const MismatchPenaltyRef mismatch_penalty = std::make_shared<MismatchPenalty>(
			m_query->mismatch_length_penalty(), document()->max_len_s());

	py::list regions;
	const int32_t n = static_cast<int32_t>(match.size());

	const int pos_filter = _pos_filter();
	std::vector<int16_t> index_map;
	if (pos_filter >= 0) {
		int16_t k = 0;
		index_map.resize(sentence().n_tokens);
		for (int32_t i = 0; i < sentence().n_tokens; i++) {
			index_map[k] = i;
			if (s_tokens.at(token_at + i).pos != pos_filter) {
				k++;
			}
		}
	}

	for (int32_t i = 0; i < n; i++) {
	    int match_at_i = match[i];

		if (match_at_i < 0) {
			continue;
		}

		if (!index_map.empty()) {
			match_at_i = index_map[match_at_i];
		}

		const auto &s = s_tokens.at(token_at + match_at_i);
		const auto &t = t_tokens.at(i);

		const int32_t idx0 = s_tokens.at(last_anchor).idx;
		if (s.idx > idx0) {

			// this is for displaying the relative (!) penalty in the UI.

			float p;

			if (last_matched) {
				p = (*mismatch_penalty.get())(token_at + match_at_i - last_anchor);
			} else {
				p = 0.0f;
			}

			regions.append(std::make_shared<Region>(
				s_text.substr(idx0, s.idx - idx0), p));
		}

#if 0
		if (document()->id() == 17 && sentence_id() == 287) {
			printf("?? %d %d\n", (int)idx0, (int)s.idx);
		}
#endif


		regions.append(std::make_shared<MatchedRegion>(
			scores[i],
			s_text.substr(s.idx, s.len),
			t_text.substr(t.idx, t.len),
			document()->vocabulary(),
			TokenRef{s_tokens_ref, token_at + match_at_i},
			TokenRef{t_tokens_ref, i},
			m_metric->origin(s.id, i)
		));

#if 0
		if (document()->id() == 17 && sentence_id() == 287) {
			printf("%f %f\n", scores[i].similarity, scores[i].weight);
			printf("match %s %s\n", s_text.substr(s.idx, s.len).c_str(), t_text.substr(t.idx, t.len).c_str());
		}
#endif

		last_anchor = token_at + match_at_i + 1;
		last_matched = true;
	}

	const int32_t up_to = std::min(last_anchor + window_size, int32_t(s_tokens.size() - 1));
	if (up_to > last_anchor) {
		const int32_t idx0 = s_tokens.at(last_anchor).idx;
		regions.append(std::make_shared<Region>(
			s_text.substr(idx0, s_tokens.at(up_to).idx - idx0)));
	}

	return regions;
}

py::list Match::omitted() const {

	const auto &t_tokens_ref = m_query->tokens();
	const std::vector<Token> &t_tokens = *t_tokens_ref.get();
	const std::string &t_text = m_query->text();

	py::list not_used;

	const auto &match = this->match();
	for (int i = 0; i < int(match.size()); i++) {
		if (match[i] < 0) {
			const auto &t = t_tokens.at(i);
			not_used.append(py::str(t_text.substr(t.idx, t.len)));
		}
	}

	return not_used;
}

class GroundTruth {
	struct Item {
		int32_t document_id;
		int32_t sentence_id;
	};

	std::vector<Item> m_items;
};

class ResultSet {
	const size_t m_max_matches;
	const float m_min_score;

public:
	ResultSet(
		const size_t p_max_matches,
		const float p_min_score) :

		m_max_matches(p_max_matches),
		m_min_score(p_min_score) {

		m_matches.reserve(p_max_matches);
	}

	inline float worst_score() const {
		if (m_matches.empty()) {
			return m_min_score;
		} else {
			return m_matches[0]->score();
		}
	}

	inline void add(const MatchRef &p_match) {

		if (p_match->score() > worst_score()) {

			if (m_matches.size() > m_max_matches) {

				std::pop_heap(
					m_matches.begin(),
					m_matches.end(),
					Match::is_worse());

				m_matches.pop_back();
			}

			m_matches.push_back(p_match);

			std::push_heap(
				m_matches.begin(),
				m_matches.end(),
				Match::is_worse());
		}
	}

	inline size_t size() const {
		return m_matches.size();
	}

	void extend(
		const ResultSet &p_set) {

		m_matches.reserve(
			m_matches.size() + p_set.m_matches.size());

		for (const auto &a : p_set.m_matches) {
			m_matches.push_back(a);

			std::push_heap(
				m_matches.begin(),
				m_matches.end(),
				Match::is_worse());
		}

		while (m_matches.size() > m_max_matches) {
				std::pop_heap(
					m_matches.begin(),
					m_matches.end(),
					Match::is_worse());

				m_matches.pop_back();
		}
	}

	void extend_and_remove_duplicates(
		const ResultSet &p_set) {

		 // not yet implemented, should remove
		// duplicate results on the same sentence.

		 PPK_ASSERT(false);
	}

	py::list best_n(size_t p_count) const;

	float precision(const GroundTruth &p_truth) const {
		return 0.0f; // to be implemented
	}

	float recall(const GroundTruth &p_truth) const {
		return 0.0f; // to be implemented
	}

private:
	const QueryRef m_query;

	// a heap such that m_matches[0] contains the
	// match with the worst/lowest score.
	std::vector<MatchRef> m_matches;

	//std::map<MatchDigest, float> m_scores;
};

struct CombineSum {
	inline float operator()(float x, float y) const {
		return x + y;
	}
};

struct CombineMin {
	inline float operator()(float x, float y) const {
		return std::min(x, y);
	}
};

struct CombineMax {
	inline float operator()(float x, float y) const {
		return std::max(x, y);
	}
};

class Matcher {
public:
	virtual ~Matcher() {
	}

	virtual void match(const ResultSetRef &p_matches) = 0;
};

typedef std::shared_ptr<Matcher> MatcherRef;

class MatcherBase : public Matcher {
protected:
	const QueryRef m_query;
	const DocumentRef m_document;
	const MetricRef m_metric;

	const MismatchPenaltyRef m_mismatch_penalty;
	const MatchRef m_no_match;

	Aligner<int16_t, float> m_aligner;

	template<typename SCORES, typename COMBINE, typename REVERSE>
	inline MatchRef optimal_match(
		const int32_t sentence_id,
		const SCORES &scores,
		const int16_t scores_variant_id,
		const COMBINE &combine,
		const float p_min_score,
		const REVERSE &reverse) {

		const int len_s = scores.s_len();
		const int len_t = m_query->len();

		if (len_t < 1 || len_s < 1) {
			return m_no_match;
		}

		const auto &gap_cost = *m_mismatch_penalty.get();

		m_aligner.waterman_smith_beyer(
			scores,
			gap_cost,
			len_s,
			len_t,
			smith_waterman_zero);

		float raw_score = m_aligner.score();

		reverse(m_aligner.mutable_match(), len_s);

		float best_final_score = m_query->normalized_score<COMBINE>(
			raw_score, m_aligner.match());

		if (best_final_score > p_min_score) {

			return std::make_shared<Match>(
				m_query,
				m_metric,
				scores_variant_id,
				MatchDigest(m_document, sentence_id, m_aligner.match()),
				best_final_score);
		} else {

			return m_no_match;
		}
	}

public:
	MatcherBase(
		const QueryRef &p_query,
		const DocumentRef &p_document,
		const MetricRef &p_metric,
		const MismatchPenaltyRef &p_mismatch_penalty) :

		m_query(p_query),
		m_document(p_document),
		m_metric(p_metric),
		m_mismatch_penalty(p_mismatch_penalty),
		m_no_match(std::make_shared<Match>(
			m_query,
			m_metric,
			-1,
			MatchDigest(m_document, -1, std::vector<int16_t>()),
			p_query->min_score()
		)),
		m_aligner(p_document->max_len_s(), m_query->len()) {
	}

};

template<typename Scores>
class ReversedScores {
	const Scores &m_scores;
	const int m_len_s;
	const int m_len_t;

public:
	inline ReversedScores(const Scores &scores, int len_t) :
		m_scores(scores), m_len_s(scores.s_len()), m_len_t(len_t) {
	}

	inline int s_len() const {
	    return m_len_s;
	}

	inline float operator()(int u, int v) const {
		return m_scores(m_len_s - 1 - u, m_len_t - 1 - v);
	}
};

void reverse_alignment(std::vector<int16_t> &match, int len_s) {
	for (size_t i = 0; i < match.size(); i++) {
		int16_t u = match[i];
		if (u >= 0) {
			match[i] = len_s - 1 - u;
		}
	}

	std::reverse(match.begin(), match.end());
}

template<typename Scores, typename Combine>
class MatcherImpl : public MatcherBase {
	const std::vector<Scores> m_scores;
	const Combine m_combine;

public:
	MatcherImpl(
		const QueryRef &p_query,
		const DocumentRef &p_document,
		const MetricRef &p_metric,
		const std::vector<Scores> &p_scores) :

		MatcherBase(
			p_query,
			p_document,
			p_metric,
			std::make_shared<MismatchPenalty>(
				p_query->mismatch_length_penalty(),
				p_document->max_len_s())
		),
		m_scores(p_scores),
		m_combine(Combine()) {

	}

	virtual void match(
		const ResultSetRef &p_matches) {

		std::vector<Scores> good_scores;
		good_scores.reserve(m_scores.size());
		for (const auto &scores : m_scores) {
			if (scores.good()) {
				good_scores.push_back(scores);
			}
		}
		if (good_scores.empty()) {
			return;
		}

		const int pos_filter = m_query->ignore_determiners() ?
		    m_document->vocabulary()->det_pos() : -1;

		const auto &sentences = m_document->sentences();
		const size_t n_sentences = sentences.size();
		//const size_t max_len_s = m_document->max_len_s();

		size_t token_at = 0;

		for (size_t sentence_id = 0;
			sentence_id < n_sentences && !m_query->aborted();
			sentence_id++) {

			const Sentence &sentence = sentences[sentence_id];
			const int len_s = sentence.n_tokens;

			if (len_s < 1) {
				continue;
			}

			MatchRef best_sentence_match = m_no_match;

			for (const auto &scores : good_scores) {

					const auto sentence_scores = scores.create_sentence_scores(
					    token_at, len_s, pos_filter);

					MatchRef m = optimal_match(
						sentence_id,
						sentence_scores,
						scores.variant(),
						m_combine,
						p_matches->worst_score(),
						[] (std::vector<int16_t> &match, int len_s) {});

					if (m_query->bidirectional()) {
						MatchRef m_reverse = optimal_match(
							sentence_id,
							ReversedScores(
    							sentence_scores, m_query->len()),
							scores.variant(),
							m_combine,
							p_matches->worst_score(),
							reverse_alignment);

						if (m_reverse->score() > m->score()) {
							m = m_reverse;
						}
					}

#if 0
					if (m_document->id() == 17 && sentence_id == 287) {
						m_matrix.print(m_query, m_document, sentence_id, m_query->len(), len_s);
						printf("match score: %.1f\n", 100.0f * m->score());

						int kk = 0;
						for (auto m : m_matrix.match) {
							if (m >= 0) {
								printf("m %d / %d\n", m, kk);
								printf("sim: %f\n", sentence_scores.similarity(m, kk));
								printf("weight: %f\n", sentence_scores.weight(m, kk));
								printf("combined: %f\n", sentence_scores(m, kk));
							}
							kk++;
						}


					}
#endif

					if (m->score() > best_sentence_match->score()) {
						best_sentence_match = m;
					}
			}

			if (best_sentence_match->score() > m_no_match->score()) {

				best_sentence_match->compute_scores(
					m_scores.at(best_sentence_match->scores_variant_id()), len_s);

#if 0
				if (m_document->id() == 17 && sentence_id == 287) {
					 best_sentence_match->print_scores();
				}
#endif

				p_matches->add(best_sentence_match);
			}

			token_at += len_s;
		}
	}
};

static void add_dummy_token(std::vector<Token> &tokens) {
		if (tokens.empty()) {
			return;
		}
		// adding a last dummy token with the correct idx is handy.
		Token t;
		t.id = -1;
		t.idx = tokens.rbegin()->idx + tokens.rbegin()->len;
		t.len = 0;
		t.pos = -1;
		t.tag = -1;
		tokens.push_back(t);
}

class Document;
typedef std::shared_ptr<Document> DocumentRef;

Document::Document(
	int64_t p_document_id,
	VocabularyRef p_vocab,
	const std::string &p_text,
	const py::object &p_sentences,
	const py::object &p_tokens,
	const py::dict &p_metadata,
	const std::string &p_cache_path = std::string()):

	m_id(p_document_id),
	m_vocab(p_vocab),
	m_text(p_text),
	m_metadata(p_metadata),
	m_cache_path(p_cache_path) {

	const auto sentences_table = unwrap_table(p_sentences);
	m_sentences = unpack_sentences(sentences_table);

	const auto tokens_table = unwrap_table(p_tokens);
	m_tokens = unpack_tokens(
		p_vocab, MODIFY_VOCABULARY, m_text, tokens_table);

	add_dummy_token(*m_tokens.get());

	size_t max_len = 0;
	for (const auto &s : m_sentences) {
		max_len = std::max(max_len, size_t(s.n_tokens));
	}
	m_max_len_s = max_len;
}

ResultSetRef Document::find(const QueryRef &p_query) {

	if (m_tokens->empty()) {
		return ResultSetRef();
	}

	py::gil_scoped_release release;
	return p_query->match(shared_from_this());
}

py::list ResultSet::best_n(size_t p_count) const {

	std::vector sorted(m_matches);
	std::sort_heap(sorted.begin(), sorted.end(), Match::is_worse());

	py::list matches;

	if (!sorted.empty()) {

		for (auto i = sorted.begin(); i != sorted.begin() + std::min(m_matches.size(), p_count); i++) {
			matches.append(*i);
		}
	}

	return matches;
}

template<typename Scores>
MatcherRef create_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document,
	const MetricRef &p_metric,
	const std::vector<Scores> &scores) {

	const auto &c = p_query->cost_combine_function();
	if (c == "min") {
		return std::make_shared<MatcherImpl<Scores, CombineMin>>(
			p_query, p_document, p_metric, scores);
	} else if (c == "max") {
		return std::make_shared<MatcherImpl<Scores, CombineMax>>(
			p_query, p_document, p_metric, scores);
	} else if (c == "sum") {
		return std::make_shared<MatcherImpl<Scores, CombineSum>>(
			p_query, p_document, p_metric, scores);
	} else {
		throw std::runtime_error(
			std::string("illegal combine function " + c));
	}
}


MatcherRef FastMetric::create_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document) {

	auto self = std::dynamic_pointer_cast<FastMetric>(shared_from_this());

	std::vector<FastScores> scores;
	scores.emplace_back(FastScores(p_query, p_document, self));

	return ::create_matcher(p_query, p_document, self, scores);
}

inline FastScores::FastScores(
    const QueryRef &p_query,
    const DocumentRef &p_document,
    const FastMetricRef &p_metric) :

    m_query(p_query),
    m_document(p_document),
    m_metric(p_metric) {

    m_filtered.resize(p_document->max_len_s());
}

inline FastSentenceScores FastScores::create_sentence_scores(
	const size_t p_s_offset,
	const size_t p_s_len,
	const int p_pos_filter) const {

	const Token *s_tokens = m_document->tokens()->data();
	const Token *t_tokens = m_query->tokens()->data();

	if (p_pos_filter > -1) {
	    const Token *s = s_tokens + p_s_offset;
	    Token *new_s = m_filtered.data();
        PPK_ASSERT(p_s_len <= m_filtered.size());

	    size_t new_s_len = 0;
        for (size_t i = 0; i < p_s_len; i++) {
            if (s[i].pos != p_pos_filter) {
                new_s[new_s_len++] = s[i];
            }
        }

        return FastSentenceScores(
            m_metric,
            new_s,
            new_s_len,
            t_tokens);
	}
    else {
        return FastSentenceScores(
            m_metric,
            s_tokens + p_s_offset,
            p_s_len,
            t_tokens);
    }
}

MetricRef ElmoEmbedding::create_metric(
		const TokenIdArray &p_vocabulary_to_embedding,
		const std::vector<float> &p_idf,
		const std::string &p_embedding_similarity,
		const std::string &p_needle_text,
		const std::vector<Token> &p_needle,
		float p_pos_mismatch_penalty,
		float p_similarity_falloff,
		float p_similarity_threshold,
		const POSWMap &p_pos_weights,
		const float p_idf_weight) {

		PPK_ASSERT(p_embedding_similarity == "cosine"); // hard-coded right now.

		return std::make_shared<ElmoMetric>(
			p_needle_text,
			p_needle);
}

MatcherRef ElmoMetric::create_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document) {

	auto self = std::dynamic_pointer_cast<ElmoMetric>(shared_from_this());

	std::vector<ElmoScores> scores;
	for (size_t i = 0; i < self->num_samples(); i++) {
			scores.emplace_back(ElmoScores(p_document, self, i));
	}

	return ::create_matcher(p_query, p_document, self, scores);
}

ResultSetRef Query::match(
	const DocumentRef &p_document) {

	ResultSetRef matches = std::make_shared<ResultSet>(
		max_matches(), min_score());

	const auto me = shared_from_this();

	for (const auto &metric : m_metrics) {
		auto matcher = metric->create_matcher(
			me,
			p_document);
		matcher->match(matches);
	}

	return matches;
}

ElmoScores::ElmoScores(
	const DocumentRef &p_document,
	const ElmoMetricRef &p_metric,
	const int p_sample) :

	m_document(p_document),
	m_metric(p_metric),
	m_sample(p_sample),
	m_document_matrix(std::make_shared<LargeMatrix>(p_document->path() + "/elmo.dat")) {

	m_good = m_document_matrix->exists();
	if (m_good) {
		m_document_matrix->open();
	}
}

py::str backend_version() {
	return "2019.09.20.1";
}

// !!!
// caution: name in PYBIND11_MODULE below needs to match filename
// !!!
PYBIND11_MODULE(vcore, m) {
	/*py::class_<Vocabulary, VocabularyRef> vocabulary(m, "Vocabulary");
	.def(py::init<const std::string &>())
	vocabulary.def("add", &Vocabulary::add);*/

	m.def("init_pyarrow", &init_pyarrow);
	m.def("backend_version", &backend_version);

	py::class_<Region, RegionRef> region(m, "Region");
	region.def_property_readonly("s", &Region::s);
	region.def_property_readonly("mismatch_penalty", &Region::mismatch_penalty);
	region.def_property_readonly("matched", &Region::is_matched);

	py::class_<MatchedRegion, Region, MatchedRegionRef> matched_region(m, "MatchedRegion");
	matched_region.def_property_readonly("t", &MatchedRegion::t);
	matched_region.def_property_readonly("similarity", &MatchedRegion::similarity);
	matched_region.def_property_readonly("weight", &MatchedRegion::weight);
	matched_region.def_property_readonly("pos_s", &MatchedRegion::pos_s);
	matched_region.def_property_readonly("pos_t", &MatchedRegion::pos_t);
	matched_region.def_property_readonly("metric", &MatchedRegion::metric);

	py::class_<Match, MatchRef> match(m, "Match");
	match.def_property_readonly("score", &Match::score);
	match.def_property_readonly("metric", &Match::metric);
	match.def_property_readonly("document", &Match::document);
	match.def_property_readonly("sentence_id", &Match::sentence_id);
	match.def_property_readonly("location", &Match::location);
	match.def_property_readonly("regions", &Match::regions);
	match.def_property_readonly("omitted", &Match::omitted);

	py::class_<Embedding, EmbeddingRef> embedding(m, "Embedding");

	py::class_<FastEmbedding, Embedding, FastEmbeddingRef> fast_embedding(m, "FastEmbedding");
	fast_embedding.def(py::init<const std::string &, py::object>());

	fast_embedding.def("add_apsynp", &FastEmbedding::add_apsynp);
	fast_embedding.def("add_nicdm", &FastEmbedding::add_nicdm);

	fast_embedding.def("cosine_similarity", &FastEmbedding::cosine_similarity);
	fast_embedding.def("similarity_matrix", &FastEmbedding::similarity_matrix);
	fast_embedding.def("load_percentiles", &FastEmbedding::load_percentiles);

	fast_embedding.def_property_readonly("n_tokens", &FastEmbedding::n_tokens);
	fast_embedding.def_property_readonly("measures", &FastEmbedding::measures);

	py::class_<ElmoEmbedding, Embedding, ElmoEmbeddingRef> elmo_embedding(m, "ElmoEmbedding");
	elmo_embedding.def(py::init());

	py::class_<Vocabulary, VocabularyRef> vocabulary(m, "Vocabulary");
	vocabulary.def(py::init());
	vocabulary.def("add_embedding", &Vocabulary::add_embedding);

	py::class_<Query, QueryRef> query(m, "Query");
	query.def(py::init<VocabularyRef, const std::string &, py::handle, py::kwargs>());
	query.def("abort", &Query::abort);

	py::class_<Document, DocumentRef> document(m, "Document");

	document.def(py::init<int64_t, VocabularyRef, const std::string&, py::object, py::object,
		py::dict, const std::string&>());
	document.def("find", &Document::find);
	document.def("__str__", &Document::__str__);
	document.def("__repr__", &Document::__str__);
	document.def_property_readonly("id", &Document::id);
	document.def_property_readonly("path", &Document::path);
	document.def_property_readonly("metadata", &Document::metadata);
	document.def_property_readonly("sentences", &Document::py_sentences_as_tokens);
	document.def_property_readonly("sentences_as_text", &Document::py_sentences_as_text);
	document.def_property_readonly("n_tokens", &Document::n_tokens);
	document.def_property_readonly("n_sentences", &Document::n_sentences);

	py::class_<ResultSet, ResultSetRef> result_set(m, "ResultSet");
	result_set.def_property_readonly("size", &ResultSet::size);
	result_set.def("best_n", &ResultSet::best_n);
	result_set.def("extend", &ResultSet::extend);

	py::class_<LargeMatrix, LargeMatrixRef> matrix(m, "LargeMatrix");
	matrix.def(py::init<const std::string &>());
	matrix.def("create", &LargeMatrix::create);
	matrix.def("close", &LargeMatrix::close);
	matrix.def("write", &LargeMatrix::write);
}
