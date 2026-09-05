import csv
import json
import os
import platform
import statistics
from typing import Dict, List

import numpy as np
import nltk
from scipy.optimize import linear_sum_assignment
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from json_handler import JsonHandler
from topic_modeling import TopicModeler


OUTPUT_DIR = "./data/output"
EVAL_DIR = "./evaluation"
TOP_N = 10
K = 10
SEEDS = [0, 21, 42, 84, 123]
ALGORITHMS = ["bertopic", "lda", "nmf", "svd"]

nltk.download("stopwords", quiet=True)
STOPWORDS = nltk.corpus.stopwords.words("spanish")


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_documents() -> List[str]:
    path = os.path.join(OUTPUT_DIR, "topic_modeling_input.json")
    documents = JsonHandler.load_topic_modeling_input(path)
    if not documents:
        raise RuntimeError("No documents found in data/output/topic_modeling_input.json")
    return [str(doc).strip() for doc in documents if str(doc).strip()]


def normalize_term(term: str) -> str:
    return str(term).strip().lower().replace(" ", "_")


def topic_number(name: str) -> int:
    try:
        return int(str(name).replace("Topic", "").strip())
    except Exception:
        return 10**9


def extract_topics(result: Dict) -> List[List[str]]:
    topics = []
    for key in sorted(result.get("topics", {}), key=topic_number):
        words = result["topics"][key].get("words", [])[:TOP_N]
        topics.append([
            normalize_term(word)
            for word in words
            if str(word).strip()
        ])
    return topics


def tokenize_for_coherence(text: str) -> List[str]:
    """
    Include both unigrams and adjacent bigrams so that the coherence corpus
    represents the same 1-2 gram vocabulary used by the original V8 models.
    """
    tokens = [tok.lower() for tok in str(text).split() if tok.strip()]
    expanded = []
    for i, token in enumerate(tokens):
        expanded.append(token)
        if i < len(tokens) - 1:
            expanded.append(f"{token}_{tokens[i + 1]}")
    return expanded


def coherence_scores(
    topics: List[List[str]],
    tokenized_documents: List[List[str]]
):
    dictionary = Dictionary(tokenized_documents)

    valid_topics = []
    for topic in topics:
        valid = [word for word in topic if word in dictionary.token2id]
        if len(valid) >= 2:
            valid_topics.append(valid)

    if not valid_topics:
        raise RuntimeError("No valid topic terms were found for coherence calculation.")

    c_v = CoherenceModel(
        topics=valid_topics,
        texts=tokenized_documents,
        dictionary=dictionary,
        coherence="c_v",
        topn=TOP_N,
        processes=1
    ).get_coherence()

    c_npmi = CoherenceModel(
        topics=valid_topics,
        texts=tokenized_documents,
        dictionary=dictionary,
        coherence="c_npmi",
        topn=TOP_N,
        processes=1
    ).get_coherence()

    return float(c_v), float(c_npmi)


def topic_diversity(topics: List[List[str]]) -> float:
    words = [word for topic in topics for word in topic[:TOP_N]]
    if not words:
        return float("nan")
    return len(set(words)) / len(words)


def jaccard(topic_a: List[str], topic_b: List[str]) -> float:
    a = set(topic_a[:TOP_N])
    b = set(topic_b[:TOP_N])
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def matched_topic_stability(
    reference_topics: List[List[str]],
    candidate_topics: List[List[str]]
) -> float:
    """
    Hungarian one-to-one matching of top-word Jaccard similarities.

    BERTopic is rerun with the original V8 adaptive/auto procedure, so a seed
    may return a different number of topics. Unmatched topics count as zero
    through division by max(number of reference topics, number of candidate topics).
    """
    if not reference_topics or not candidate_topics:
        return 0.0

    matrix = np.zeros(
        (len(reference_topics), len(candidate_topics)),
        dtype=float
    )

    for i, ref_topic in enumerate(reference_topics):
        for j, cand_topic in enumerate(candidate_topics):
            matrix[i, j] = jaccard(ref_topic, cand_topic)

    rows, cols = linear_sum_assignment(-matrix)
    matched_sum = float(matrix[rows, cols].sum())
    denominator = max(len(reference_topics), len(candidate_topics))

    return matched_sum / denominator if denominator else 0.0


def load_main_results(documents: List[str]) -> Dict[str, Dict]:
    """
    Uses the existing saved BERTopic manuscript result.
    LDA/NMF/SVD are expected to be freshly generated at K=10.
    """
    results = {}

    for algorithm in ALGORITHMS:
        path = os.path.join(
            OUTPUT_DIR,
            f"topic_results_{algorithm}.json"
        )

        if not os.path.exists(path):
            raise RuntimeError(
                f"Missing {path}. "
                "Keep the existing BERTopic file and run LDA, NMF and SVD/LSA at K=10 first."
            )

        result = load_json(path)
        topics = extract_topics(result)

        if len(topics) != K:
            raise RuntimeError(
                f"{algorithm.upper()} currently has {len(topics)} topics. "
                f"The evaluation requires K={K} main topics for every model."
            )

        doc_count = result.get("metadata", {}).get("document_count")
        if doc_count is not None and doc_count != len(documents):
            raise RuntimeError(
                f"{algorithm.upper()} result used {doc_count} documents but "
                f"the manuscript corpus contains {len(documents)}."
            )

        results[algorithm] = result

    return results


def run_stability(
    documents: List[str],
    main_results: Dict[str, Dict]
) -> Dict[str, Dict]:
    stability = {}

    for algorithm in ALGORITHMS:
        reference_topics = extract_topics(main_results[algorithm])

        seed_scores = []
        seed_topic_counts = {}

        for seed in SEEDS:
            print(f"Stability run: {algorithm.upper()} seed={seed}")

            # Preserve the original V8 BERTopic procedure for stability:
            # adaptive HDBSCAN + nr_topics='auto'.
            requested_topics = "auto" if algorithm == "bertopic" else K

            result = TopicModeler.extract_topics(
                documents,
                algorithm=algorithm,
                n_topics=requested_topics,
                n_words=TOP_N,
                stopwords=STOPWORDS,
                random_state=seed
            )

            candidate_topics = extract_topics(result)
            seed_topic_counts[str(seed)] = len(candidate_topics)

            score = matched_topic_stability(
                reference_topics,
                candidate_topics
            )
            seed_scores.append(score)

        stability[algorithm] = {
            "seeds": SEEDS,
            "topic_counts": seed_topic_counts,
            "scores_against_saved_main_result": seed_scores,
            "mean": float(statistics.mean(seed_scores)),
            "sd": (
                float(statistics.stdev(seed_scores))
                if len(seed_scores) > 1
                else 0.0
            )
        }

    return stability


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    documents = load_documents()
    print(f"Loaded manuscript topic corpus: {len(documents)} documents")

    tokenized_documents = [
        tokenize_for_coherence(document)
        for document in documents
    ]

    main_results = load_main_results(documents)

    metrics = {}

    for algorithm in ALGORITHMS:
        topics = extract_topics(main_results[algorithm])

        c_v, c_npmi = coherence_scores(
            topics,
            tokenized_documents
        )

        metrics[algorithm] = {
            "model": (
                "BERTopic"
                if algorithm == "bertopic"
                else "SVD"
                if algorithm == "svd"
                else algorithm.upper()
            ),
            "k": len(topics),
            "c_v": c_v,
            "c_npmi": c_npmi,
            "diversity": topic_diversity(topics),
            "outliers": (
                main_results[algorithm]
                .get("diagnostics", {})
                .get("outlier_percentage")
                if algorithm == "bertopic"
                else None
            )
        }

    stability = run_stability(
        documents,
        main_results
    )

    for algorithm in ALGORITHMS:
        metrics[algorithm]["stability"] = stability[algorithm]["mean"]
        metrics[algorithm]["stability_sd"] = stability[algorithm]["sd"]

    print("\nTopic-model evaluation")
    print(
        f"{'model':<10} {'c_v':>8} {'c_npmi':>9} "
        f"{'diversity':>11} {'stability':>12} {'outliers':>10}"
    )

    for algorithm in ALGORITHMS:
        row = metrics[algorithm]
        outlier_display = (
            "-"
            if row["outliers"] is None
            else f"{row['outliers']:.1f}%"
        )

        print(
            f"{row['model']:<10} "
            f"{row['c_v']:>8.4f} "
            f"{row['c_npmi']:>9.4f} "
            f"{row['diversity']:>11.4f} "
            f"{row['stability']:>12.4f} "
            f"{outlier_display:>10}"
        )

    output = {
        "design": {
            "main_topic_count": K,
            "top_n_words": TOP_N,
            "stability_seeds": SEEDS,
            "bertopic_main_result": (
                "existing manuscript BERTopic result; not regenerated"
            ),
            "bertopic_stability": (
                "original V8 adaptive BERTopic procedure rerun in memory"
            ),
            "classical_stability": (
                "LDA/NMF/SVD rerun at fixed K=10"
            ),
            "stability_metric": (
                "Hungarian-matched top-10-word Jaccard similarity "
                "against each model's saved main result"
            ),
            "diversity_metric": (
                "unique top-10 topic terms / total top-10 topic terms"
            )
        },
        "metrics": metrics,
        "stability_details": stability,
        "software": {
            "python": platform.python_version()
        }
    }

    json_path = os.path.join(
        EVAL_DIR,
        "topic_model_evaluation.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(
        EVAL_DIR,
        "topic_model_evaluation.csv"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "k",
                "c_v",
                "c_npmi",
                "diversity",
                "stability",
                "stability_sd",
                "outliers"
            ]
        )
        writer.writeheader()
        for algorithm in ALGORITHMS:
            writer.writerow(metrics[algorithm])

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
