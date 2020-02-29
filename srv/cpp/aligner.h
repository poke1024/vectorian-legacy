// Author: Bernhard Liebl, 2020
// Released under a MIT license.

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// Aligner always computes one best alignment, but there
// might be multiple such alignments.

template<
	typename Index=int16_t,
	typename SimilarityScore=float>

class Aligner {
private:
	class Fold {
	private:
		SimilarityScore _score;
		std::pair<Index, Index> _traceback;
	
	public:
		inline Fold(SimilarityScore zero_score) :
			_score(zero_score),
			_traceback(std::make_pair(-1, -1)) {
		}

		inline Fold(
			SimilarityScore score,
			const std::pair<Index, Index> &traceback) :
			
			_score(score),
			_traceback(traceback) {

		}

		inline void update(
			const SimilarityScore score,
			const std::pair<Index, Index> &traceback) {

			if (score > _score) {
				_score = score;
				_traceback = traceback;
			}
		}

		inline SimilarityScore score() const {
			return _score;
		}

		inline const std::pair<Index, Index> &traceback() const {
			return _traceback;
		}
	};

	const size_t _max_len_s;
	const size_t _max_len_t;

	Eigen::Tensor<SimilarityScore, 2> _values;
	Eigen::Tensor<std::pair<Index, Index>, 2> _traceback;

	SimilarityScore _best_score;
	std::vector<Index> _best_match;

	inline void reconstruct_local_alignment(
		const Index len_t,
		const Index len_s,
		const SimilarityScore zero_similarity) {

		const auto &values = _values;
		const auto &traceback = _traceback;

		SimilarityScore score = values(0, 0);
		Index best_u = 0, best_v = 0;

		for (Index v = 0; v < len_t; v++) {
			for (Index u = 0; u < len_s; u++) {
				const SimilarityScore s = values(u, v);
				if (s > score) {
					score = s;
					best_u = u;
					best_v = v;
				}
			}
		}

		_best_score = score;

		_best_match.resize(len_t);
		std::fill(_best_match.begin(), _best_match.end(), -1);

		Index u = best_u;
		Index v = best_v;
		while (u >= 0 && v >= 0 && values(u, v) > zero_similarity) {
			_best_match[v] = u;
			std::tie(u, v) = traceback(u, v);
		}
	}

	inline void reconstruct_global_alignment(
		const Index len_t,
		const Index len_s) {

		_best_match.resize(len_t);
		std::fill(_best_match.begin(), _best_match.end(), -1);

		Index u = len_s - 1;
		Index v = len_t - 1;
		_best_score = _values(u, v);

		while (u >= 0 && v >= 0) {
			_best_match[v] = u;
			std::tie(u, v) = _traceback(u, v);
		}
	}

public:
	Aligner(Index max_len_s, Index max_len_t) :
		_max_len_s(max_len_s),
		_max_len_t(max_len_t) {

		_values.resize(max_len_s, max_len_t);
		_traceback.resize(max_len_s, max_len_t);
		_best_match.reserve(max_len_t);		
	}

	inline SimilarityScore score() const {
		return _best_score;
	}

	inline const std::vector<Index> &match() const {
		return _best_match;
	}

	inline std::vector<Index> &mutable_match() {
		return _best_match;
	}

#if !defined(ALIGNER_SLIM)
	std::string pretty_printed(
		const std::string &s,
		const std::string &t) {

		std::ostringstream out[3];

		int i = 0;
		for (int j = 0; j < t.length(); j++) {
			auto m = _best_match[j];
			if (m < 0) {
				out[0] << "-";
				out[1] << " ";
				out[2] << t[j];
			} else {
				while (i < m) {
					out[0] << s[i];
					out[1] << " ";
					out[2] << "-";
					i += 1;
				}

				out[0] << s[m];
				out[1] << "|";
				out[2] << t[j];
				i = m + 1;
			}
		}

		while (i < s.length()) {
			out[0] << s[i];
			out[1] << " ";
			out[2] << "-";
			i += 1;
		}

		std::ostringstream r;
		r << out[0].str() << "\n" << out[1].str() << "\n" << out[2].str();
		return r.str();
	}
#endif

	template<typename Similarity>
	void needleman_wunsch(
		const Similarity &similarity,
		const SimilarityScore gap_cost, // linear
		const Index len_s,
		const Index len_t) {

		if (len_t < 1 || len_s < 1) {
			throw std::invalid_argument("len must be >= 1");
		}

		if (size_t(len_t) > _max_len_t || size_t(len_s) > _max_len_s) {
			throw std::invalid_argument("len larger than max");			
		}

		auto &values = _values;
		auto &traceback = _traceback;

		const auto nwvalues = [&values, &gap_cost] (Index u, Index v) {
			if (u >= 0 && v >= 0) {
				return values(u, v);
			} else if (u < 0) {
				return -gap_cost * (v + 1);
			} else {
				return -gap_cost * (u + 1);
			}
		};

		for (Index u = 0; u < len_s; u++) {

			for (Index v = 0; v < len_t; v++) {

				const SimilarityScore s0 =
					nwvalues(u - 1, v - 1);
				const SimilarityScore s1 =
					similarity(u, v);
				Fold best(
					s0 + s1,
					std::make_pair(u - 1, v - 1));

				best.update(
					nwvalues(u - 1, v) - gap_cost,
					std::make_pair(u - 1, v));

				best.update(
					nwvalues(u, v - 1) - gap_cost,
					std::make_pair(u, v - 1));

				values(u, v) = best.score();
				traceback(u, v) = best.traceback();
			}
		}

		reconstruct_global_alignment(len_t, len_s);
	}

	template<typename Similarity>
	void smith_waterman(
		const Similarity &similarity,
		const SimilarityScore gap_cost, // linear
		const Index len_s,
		const Index len_t,
		const SimilarityScore zero_similarity = 0) {

		if (len_t < 1 || len_s < 1) {
			throw std::invalid_argument("len must be >= 1");
		}

		if (size_t(len_t) > _max_len_t || size_t(len_s) > _max_len_s) {
			throw std::invalid_argument("len larger than max");			
		}

		auto &values = _values;
		auto &traceback = _traceback;

		for (Index u = 0; u < len_s; u++) {

			for (Index v = 0; v < len_t; v++) {

				Fold best(zero_similarity);

				{
					const SimilarityScore s0 =
						(v > 0 && u > 0) ? values(u - 1, v - 1) : 0;
					const SimilarityScore s1 =
						similarity(u, v);
					best.update(
						s0 + s1,
						std::make_pair(u - 1, v - 1));
				}

				if (u > 0) {
					best.update(
						values(u - 1, v) - gap_cost,
						std::make_pair(u - 1, v));
				}

				if (v > 0) {
					best.update(
						values(u, v - 1) - gap_cost,
						std::make_pair(u, v - 1));
				}

				values(u, v) = best.score();
				traceback(u, v) = best.traceback();
			}
		}

		reconstruct_local_alignment(len_t, len_s, zero_similarity);
	}

	template<typename Similarity, typename Gap>
	void waterman_smith_beyer(
		const Similarity &similarity,
		const Gap &gap_cost,
		const Index len_s,
		const Index len_t,
		const SimilarityScore zero_similarity = 0) {

		if (len_t < 1 || len_s < 1) {
			throw std::invalid_argument("len must be >= 1");
		}

		if (size_t(len_t) > _max_len_t || size_t(len_s) > _max_len_s) {
			throw std::invalid_argument("len larger than max");			
		}

		auto &values = _values;
		auto &traceback = _traceback;

		for (Index u = 0; u < len_s; u++) {

			for (Index v = 0; v < len_t; v++) {

				Fold best(zero_similarity);

				{
					const SimilarityScore s0 =
						(v > 0 && u > 0) ? values(u - 1, v - 1) : 0;
					const SimilarityScore s1 =
						similarity(u, v);
					best.update(
						s0 + s1,
						std::make_pair(u - 1, v - 1));
				}

				for (Index k = 0; k < u; k++) {
					best.update(
						values(k, v) - gap_cost(u - k),
						std::make_pair(k, v));
				}

				for (Index k = 0; k < v; k++) {
					best.update(
						values(u, k) - gap_cost(v - k),
						std::make_pair(u, k));
				}

				values(u, v) = best.score();
				traceback(u, v) = best.traceback();
			}
		}

		reconstruct_local_alignment(len_t, len_s, zero_similarity);
	}
};
