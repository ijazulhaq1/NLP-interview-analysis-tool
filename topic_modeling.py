from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Tuple
import os
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from collections import defaultdict
import logging
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class TopicModeler:

    @staticmethod
    def _clean_texts(texts: List[str]) -> List[str]:
        """Pre-clean texts before processing"""
        cleaned = []
        for text in texts:
            # Remove standalone numbers and special characters
            text = ' '.join([word for word in text.split() if not word.isdigit()])
            text = re.sub(r'[^\w\sáéíóúÁÉÍÓÚñÑ]', '', text)
            cleaned.append(text)
        return cleaned

    @staticmethod
    def _join_ngrams(words: List[str], scores: List[float]) -> Tuple[List[str], List[float]]:
        """Join multi-word terms with underscores and keep highest scoring version"""
        word_dict = {}
        for word, score in zip(words, scores):
            # Replace spaces with underscores for multi-word terms
            processed_word = word.replace(' ', '_')
            if processed_word not in word_dict or score > word_dict[processed_word]:
                word_dict[processed_word] = score
        return list(word_dict.keys()), list(word_dict.values())

    @staticmethod
    def extract_topics(
            texts: List[str],
            algorithm: str = "bertopic",
            n_topics: Optional[Union[int, str]] = "auto",
            n_words: int = 10,
            stopwords: List[str] = None,
            min_cluster_size: Optional[int] = None,
            min_samples: Optional[int] = None,
            embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
            random_state: int = 42,
            group_name: str = ""
    ) -> Dict:
        """Unified topic extraction method"""
        if algorithm not in ["bertopic", "lda", "nmf", "svd"]:
            raise ValueError("Supported algorithms: bertopic, lda, nmf, svd")

        start_time = datetime.now()
        texts = [str(t).strip() for t in texts if str(t).strip()]
        texts = TopicModeler._clean_texts(texts)

        if len(texts) < 10:
            raise ValueError(f"Need ≥10 documents, got {len(texts)}")

        if algorithm == "bertopic":
            results = TopicModeler._extract_with_bertopic(
                texts=texts,
                n_topics=n_topics,
                n_words=n_words,
                stopwords=stopwords,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                embedding_model=embedding_model,
                random_state=random_state,
                group_name=group_name,
                start_time=start_time
            )
        else:
            results = TopicModeler._extract_with_sklearn(
                texts=texts,
                algorithm=algorithm,
                n_topics=n_topics,
                n_words=n_words,
                stopwords=stopwords,
                random_state=random_state,
                group_name=group_name,
                start_time=start_time
            )

        # Process words to join n-grams with underscores
        for topic_name, topic_data in results.get("topics", {}).items():
            words, scores = TopicModeler._join_ngrams(topic_data["words"], topic_data["scores"])
            results["topics"][topic_name]["words"] = words
            results["topics"][topic_name]["scores"] = scores

        # Add scope information to results
        if "block_" in group_name:
            results["scope"] = "question_block"
        elif "q_" in group_name:
            results["scope"] = "individual_question"
        else:
            results["scope"] = "all_interviews"

        return results

    @staticmethod
    def _extract_with_bertopic(
            texts: List[str],
            n_topics: Union[int, str],
            n_words: int,
            stopwords: List[str],
            min_cluster_size: Optional[int],
            min_samples: Optional[int],
            embedding_model: str,
            random_state: int,
            group_name: str,
            start_time: datetime
    ) -> Dict:
        """BERTopic implementation"""
        doc_count = len(texts)
        min_cluster_size = min_cluster_size or max(10, min(50, int(doc_count ** 0.5)))
        min_samples = min_samples or max(3, int(min_cluster_size * 0.3))

        umap_model = UMAP(
            n_neighbors=min(15, max(5, int(doc_count ** 0.25))),
            n_components=min(15, max(5, int(np.log(doc_count) * 2))),
            min_dist=0.1,  # Increased for better separation
            metric='cosine',
            random_state=random_state
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='leaf',
            prediction_data=True,
            gen_min_span_tree=True
        )

        vectorizer = CountVectorizer(
            stop_words=stopwords,
            ngram_range=(1, 2),  # Reduced from (1,3) to (1,2)
            min_df=max(2, int(doc_count * 0.001)),
            max_df=0.85,
            token_pattern=r'(?u)\b[^\d\W][^\d\W]+\b'  # Exclude pure numbers
        )

        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=SentenceTransformer(embedding_model),
            vectorizer_model=vectorizer,
            top_n_words=n_words,
            nr_topics=n_topics,
            verbose=False
        )

        with tqdm(total=3, desc=f"Processing {group_name or 'documents'}") as pbar:
            embeddings = topic_model.embedding_model.encode(texts, show_progress_bar=False)
            pbar.update(1)

            topics, _ = topic_model.fit_transform(texts, embeddings)
            pbar.update(1)

            if n_topics == "auto":
                topic_model.reduce_topics(texts, nr_topics="auto")
            pbar.update(1)

        results = TopicModeler._format_results(
            model=topic_model,
            n_words=n_words,
            group_name=group_name,
            algorithm="bertopic"
        )

        results.update({
            "diagnostics": TopicModeler._get_diagnostics(topic_model, texts, embeddings, start_time),
            "content_type": "question_block" if "block_" in group_name else "full_analysis"
        })

        return results

    @staticmethod
    def _extract_with_sklearn(
            texts: List[str],
            algorithm: str,
            n_topics: int,
            n_words: int,
            stopwords: List[str],
            random_state: int,
            group_name: str,
            start_time: datetime
    ) -> Dict:
        """Sklearn-based topic models (LDA/NMF/SVD)"""
        # Keep the original V8 TF-IDF pipeline unchanged for LDA and NMF.
        vectorizer = TfidfVectorizer(
            stop_words=stopwords,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            token_pattern=r'(?u)\b[^\d\W][^\d\W]+\b'
        )

        tfidf = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        if algorithm == "lda":
            model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=random_state,
                learning_method='online'
            )
            transformed = model.fit_transform(tfidf)

        elif algorithm == "nmf":
            model = NMF(
                n_components=n_topics,
                random_state=random_state,
                init='nndsvd'
            )
            transformed = model.fit_transform(tfidf)

        else:  # SVD / LSA
            model = TruncatedSVD(
                n_components=n_topics,
                random_state=random_state
            )
            transformed = model.fit_transform(tfidf)

        assignments = np.abs(transformed).argmax(axis=1)

        topics = {}
        for idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[:-n_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics[f"Topic {idx}"] = {
                "words": top_words,
                "scores": topic[top_words_idx].tolist(),
                "count": int((assignments == idx).sum()),
                "group": group_name
            }

        return {
            "topics": topics,
            "content_type": "question_block" if "block_" in group_name else "full_analysis",
            "algorithm": algorithm,
            "group": group_name,
            "diagnostics": {
                "processing_time": str(datetime.now() - start_time),
                "n_topics": n_topics,
                "random_state": random_state,
                "matrix": "tfidf"
            }
        }

    @staticmethod
    def _format_results(
            model,
            n_words: int,
            group_name: str,
            algorithm: str
    ) -> Dict:
        """Standardized result formatting"""
        if algorithm == "bertopic":
            topic_info = model.get_topic_info()
            topics = {}

            for _, row in topic_info.iterrows():
                if row.Topic == -1:
                    continue

                topic_words = model.get_topic(row.Topic)
                topics[f"Topic {row.Topic}"] = {
                    "words": [w for w, _ in topic_words[:n_words]],
                    "scores": [s for _, s in topic_words[:n_words]],
                    "count": int(row.Count),
                    "representative_docs": model.get_representative_docs(row.Topic)[:3],
                    "group": group_name
                }

            return {
                "topics": topics,
                "algorithm": "bertopic",  # This line already exists
                "content_type": "question_block" if "block_" in group_name else "full_analysis",
                "group": group_name
            }
        else:
            raise ValueError("Unsupported algorithm for formatting")

    @staticmethod
    def _get_diagnostics(model, texts, embeddings, start_time) -> Dict:
        """Calculate performance metrics"""
        topic_info = model.get_topic_info()
        outlier_pct = (topic_info[topic_info.Topic == -1]['Count'].sum() / len(texts)) * 100

        nbrs = NearestNeighbors(n_neighbors=5).fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)

        return {
            "processing_time": str(datetime.now() - start_time),
            "n_topics": len(topic_info) - 1,
            "outlier_percentage": round(outlier_pct, 1),
            "avg_5nn_distance": round(np.mean(distances[:, 4]), 3),
            "median_topic_size": topic_info[topic_info.Topic != -1]['Count'].median(),
            "topic_size_distribution": topic_info['Count'].value_counts().to_dict()
        }

    @staticmethod
    def generate_wordclouds(topic_data: Dict, output_path: str):
        """Generate word clouds for topics"""
        os.makedirs(output_path, exist_ok=True)

        for topic_name, data in topic_data.get("topics", {}).items():
            if not data.get("words"):
                continue

            # Process words to replace underscores with spaces for display
            display_words = [w.replace('_', ' ') for w in data["words"]]
            word_weights = dict(zip(display_words, data["scores"]))

            wc = WordCloud(
                width=1000, height=600,
                background_color='white',
                colormap='viridis',
                max_words=50,
                collocations=False  # Prevent word repetitions
            ).generate_from_frequencies(word_weights)

            plt.figure(figsize=(10, 6))
            plt.imshow(wc, interpolation='bilinear')
            plt.title(f"Topic: {topic_name}", pad=20)
            plt.axis("off")

            filename = f"{topic_name.lower().replace(' ', '_')}.png"
            plt.savefig(
                os.path.join(output_path, filename),
                bbox_inches='tight',
                dpi=150
            )
            plt.close()