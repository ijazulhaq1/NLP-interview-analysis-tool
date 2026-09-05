import os
import numpy as np
import pandas as pd
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from wordcloud import WordCloud
from bertopic import BERTopic
from tqdm import tqdm
import json
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TopicSentimentAnalyzer:
    """Handles sentiment analysis for topics"""

    @staticmethod
    def get_topic_sentiment_distribution(topic_results: Dict, sentiment_results: Dict,
                                         min_word_overlap: int = 1) -> Dict:
        distributions = defaultdict(lambda: defaultdict(int))

        for topic_name, topic_data in topic_results.get("topics", {}).items():
            topic_words = set(word.lower().strip(".,!?") for word in topic_data['words'])

            for response in sentiment_results.values():
                sentences = []
                if 'sentences' in response:
                    sentences = response['sentences']
                elif isinstance(response, dict):
                    for q_data in response.values():
                        if isinstance(q_data, dict):
                            sentences.extend(q_data.get('sentences', []))

                for sentence in sentences:
                    if isinstance(sentence, dict):
                        sentence_words = set(word.lower().strip(".,!?") for word in sentence['text'].split())
                        if len(topic_words & sentence_words) >= min_word_overlap:
                            score = round(sentence['sentiment'].get('score', 3))
                            distributions[topic_name][score] += 1
        return distributions

    @staticmethod
    def analyze_topic_sentiment(topic_results: Dict, sentiment_results: Dict, min_word_overlap: int = 1) -> Dict:
        topic_sentiments = {}

        for topic_name, topic_data in topic_results.get("topics", {}).items():
            topic_words = set(word.lower().strip(".,!?") for word in topic_data['words'])
            total_score = count = 0
            matched_sentences = []

            for response in sentiment_results.values():
                sentences = []
                if 'sentences' in response:
                    sentences = response['sentences']
                elif isinstance(response, dict):
                    for q_data in response.values():
                        if isinstance(q_data, dict):
                            sentences.extend(q_data.get('sentences', []))

                for sentence in sentences:
                    if isinstance(sentence, dict):
                        sentence_words = set(word.lower().strip(".,!?") for word in sentence['text'].split())
                        if len(topic_words & sentence_words) >= min_word_overlap:
                            total_score += sentence['sentiment'].get('score', 3)
                            count += 1
                            matched_sentences.append({
                                'text': sentence['text'],
                                'score': sentence['sentiment'].get('score', 3)
                            })

            avg_score = total_score / count if count else 3.0
            sentiment = "NEUTRAL"
            if avg_score > 3.5:
                sentiment = "POSITIVE"
            elif avg_score < 2.5:
                sentiment = "NEGATIVE"

            topic_sentiments[topic_name] = {
                'average_score': round(avg_score, 2),
                'sentiment': sentiment,
                'sample_size': count,
                'matched_sentences': matched_sentences[:5]
            }
        return topic_sentiments


class Visualizer:
    """Complete visualization class with all methods"""

    def __init__(self, output_path: str = "./data/visualizations"):
        self.output_path = output_path
        self._initialize_directories()
        self._load_nlp_model()
        self._set_styles()
        self.palettes = self._initialize_palettes()

    def _initialize_directories(self):
        algorithms = ["bertopic", "lda", "nmf", "svd"]
        for algo in algorithms:
            os.makedirs(os.path.join(self.output_path, f"{algo}_visualizations", "wordclouds"), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, f"{algo}_visualizations", "barcharts"), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, f"{algo}_visualizations", "sentiment"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "sentiment"), exist_ok=True)

    def _load_nlp_model(self):
        try:
            self.nlp = spacy.load("es_core_news_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"])
            self.nlp = spacy.load("es_core_news_sm")

    def _set_styles(self):
        sns.set_style("whitegrid", {'grid.linestyle': ':'})
        plt.rcParams.update({
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.facecolor': (0, 0, 0, 0)
        })

    def _initialize_palettes(self):
        return {
            'bertopic': sns.color_palette("viridis"),
            'lda': sns.color_palette("rocket"),
            'nmf': sns.color_palette("mako"),
            'svd': sns.color_palette("flare"),
            'sentiment': {
                'negative': '#ff6b6b',
                'neutral': '#ffd166',
                'positive': '#06d6a0',
                'confused': '#5e548e',
                'clear': '#b8bedd'
            }
        }

    def generate_all_visualizations(self, topic_data: Dict, sentiment_results: Dict):
        print("\n📊 Starting visualization generation...")

        if not sentiment_results:
            print("⚠️ No sentiment results provided")
            return

        try:
            if not topic_data:
                print("⚠️ No topic data provided")
                return

            if "algorithm" not in topic_data:
                topic_data["algorithm"] = topic_data.get("metadata", {}).get("algorithm", "bertopic")

            self._process_single_topic_data(topic_data, sentiment_results)
        except Exception as e:
            print(f"⚠️ Failed to process topic data: {str(e)}")

        self._generate_general_sentiment_visualizations(sentiment_results)
        print(f"\n🎉 All visualizations saved under {self.output_path}")

    def _process_single_topic_data(self, topic_data: Dict, sentiment_results: Dict):
        """Process a single topic data set and generate all visualizations.

        Args:
            topic_data: Dictionary containing topic modeling results
            sentiment_results: Dictionary containing sentiment analysis results
        """
        if not topic_data:
            print("⚠️ No topic data provided")
            return

        # Get algorithm from topic_data (default to bertopic if not specified)
        algo = topic_data.get("algorithm", "bertopic").lower()
        print(f"\n🔍 Generating {algo.upper()} visualizations...")

        # Generate topic visualizations (wordclouds and barcharts)
        topic_viz = self._generate_topic_visualizations(topic_data, algo)
        print(f"  ✅ Generated {len(topic_viz['wordclouds'])} word clouds")
        print(f"  ✅ Generated {len(topic_viz['barcharts'])} bar charts")

        # Generate BERTopic-specific visualizations if applicable
        if algo == "bertopic":
            bert_viz = self._generate_bertopic_specific(topic_data)
            print(f"  ✅ Generated {len(bert_viz)} BERTopic-specific visualizations")

        # Generate topic-sentiment relationship visualizations
        sentiment_viz = self._plot_topic_sentiment_relations(topic_data, sentiment_results, algo)
        print(f"  ✅ Generated {len(sentiment_viz)} topic-sentiment visualizations")

        # Generate combined topic-sentiment visualization
        combined_path = self._generate_combined_topic_sentiment_visualization(
            topic_data,
            sentiment_results,
            algo
        )
        if combined_path:
            print(f"  ✅ Generated combined topic-sentiment visualization: {combined_path}")

        # Generate general sentiment visualizations
        self._generate_general_sentiment_visualizations(sentiment_results, algo)

    def _generate_topic_visualizations(self, topic_data: Dict, algorithm: str) -> Dict:
        return {
            'wordclouds': self._generate_wordclouds(topic_data, algorithm),
            'barcharts': self._generate_topic_barcharts(topic_data, algorithm),
            'bertopic_specific': self._generate_bertopic_specific(topic_data) if algorithm == "bertopic" else []
        }

    def _generate_wordclouds(self, topic_data: Dict, algorithm: str) -> List[str]:
        paths = []
        topics = topic_data.get("topics", {})
        output_dir = os.path.join(self.output_path, f"{algorithm}_visualizations", "wordclouds")

        for topic_name, data in topics.items():
            try:
                if not data.get("words"):
                    continue

                # Deduplicate words by keeping the highest score version
                word_weights = {}
                for word, score in zip(data["words"], data["scores"]):
                    base_word = word.split()[0].lower()  # Take base form
                    if base_word not in word_weights or score > word_weights[base_word]:
                        word_weights[base_word] = score

                wc = WordCloud(
                    width=1200, height=600,
                    background_color='white',
                    colormap='viridis',
                    max_words=50,
                    collocations=False  # Prevent word repetitions
                ).generate_from_frequencies(word_weights)

                plt.figure(figsize=(12, 6))
                plt.imshow(wc, interpolation='bilinear')
                plt.title(f"{algorithm.upper()} Topic: {topic_name}", pad=20)
                plt.axis("off")

                filename = f"{algorithm}_wordcloud_{topic_name.lower().replace(' ', '_')}.png"
                path = os.path.join(output_dir, filename)
                plt.savefig(path, bbox_inches='tight', dpi=150)
                paths.append(path)
                plt.close()
            except Exception as e:
                print(f"⚠️ Failed to generate wordcloud for {topic_name}: {str(e)}")
        return paths

    def _generate_topic_barcharts(self, topic_data: Dict, algorithm: str) -> List[str]:
        paths = []
        topics = topic_data.get("topics", {})
        output_dir = os.path.join(self.output_path, f"{algorithm}_visualizations", "barcharts")
        palette = self.palettes.get(algorithm, self.palettes['bertopic'])

        for topic_name, data in topics.items():
            try:
                df = pd.DataFrame({
                    'Terms': data["words"][:10][::-1],
                    'Importance': data["scores"][:10][::-1]
                })

                plt.figure(figsize=(10, 5))
                bars = plt.barh(
                    df['Terms'], df['Importance'],
                    color=palette,
                    edgecolor='white'
                )

                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                             f'{width:.2f}', va='center', ha='left', fontsize=9)

                plt.title(f'{algorithm.upper()} Topic: {topic_name}\nKey Terms', pad=15)
                plt.xlabel('TF-IDF Score' if algorithm != "bertopic" else 'Topic Score')
                plt.grid(axis='x', alpha=0.2)
                sns.despine(left=True)

                filename = f"{algorithm}_barchart_{topic_name.lower().replace(' ', '_')}.png"
                path = os.path.join(output_dir, filename)
                plt.savefig(path, bbox_inches='tight')
                paths.append(path)
                plt.close()
            except Exception as e:
                print(f"⚠️ Failed to generate barchart for {topic_name}: {str(e)}")
        return paths

    def _generate_bertopic_specific(self, topic_data: Dict) -> List[str]:
        paths = []
        model = topic_data.get("model")
        if not isinstance(model, BERTopic):
            return paths

        try:
            output_dir = os.path.join(self.output_path, "bertopic_visualizations", "topic_hierarchies")
            os.makedirs(output_dir, exist_ok=True)

            visualizations = [
                ("hierarchy", model.visualize_hierarchy),
                ("heatmap", model.visualize_heatmap),
                ("term_rank", model.visualize_term_rank),
                ("topics", model.visualize_topics)
            ]

            for name, viz_func in visualizations:
                fig = viz_func()
                path = os.path.join(output_dir, f"bertopic_{name}.png")
                fig.write_image(path)
                paths.append(path)

        except Exception as e:
            print(f"⚠️ Failed to generate BERTopic visualizations: {str(e)}")
        return paths

    def _plot_topic_sentiment_relations(self, topic_results: Dict, sentiment_results: Dict, algorithm: str) -> List[
        str]:
        algorithm = algorithm.lower()
        output_dir = os.path.join(self.output_path, f"{algorithm}_visualizations", "sentiment")
        paths = []

        # 1. Topic Sentiment Distribution
        sentiment_dist = TopicSentimentAnalyzer.get_topic_sentiment_distribution(topic_results, sentiment_results)
        dist_path = self._plot_sentiment_distribution(sentiment_dist, algorithm, output_dir)
        if dist_path:
            paths.append(dist_path)

        # 2. Topic Sentiment Analysis
        topic_sentiments = TopicSentimentAnalyzer.analyze_topic_sentiment(topic_results, sentiment_results)
        sentiment_path = self._plot_topic_sentiment(topic_sentiments, algorithm, output_dir)
        if sentiment_path:
            paths.append(sentiment_path)

        return paths

    def _plot_sentiment_distribution(self, sentiment_dist: Dict, algorithm: str, output_dir: str) -> Optional[str]:
        try:
            data = []
            for topic, scores in sentiment_dist.items():
                for score, count in scores.items():
                    data.append({'Topic': topic, 'Score': score, 'Count': count})

            if not data:
                return None

            df = pd.DataFrame(data)
            total_counts = df.groupby('Topic')['Count'].sum()
            df['Percentage'] = df.apply(lambda x: (x['Count'] / total_counts[x['Topic']]) * 100, axis=1)

            score_colors = {
                1: self.palettes['sentiment']['negative'],
                2: self.palettes['sentiment']['negative'],
                3: self.palettes['sentiment']['neutral'],
                4: self.palettes['sentiment']['positive'],
                5: self.palettes['sentiment']['positive']
            }

            plt.figure(figsize=(12, 8))
            ax = sns.barplot(
                data=df,
                x='Topic',
                y='Percentage',
                hue='Score',
                palette=score_colors
            )

            plt.title(f'{algorithm.upper()} Topics - Sentiment Distribution', pad=20)
            plt.xlabel('Topic')
            plt.ylabel('Percentage')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Sentiment Score')
            plt.tight_layout()

            path = os.path.join(output_dir, f"{algorithm}_topic_sentiment_distribution.png")
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            return path
        except Exception as e:
            print(f"⚠️ Failed to generate sentiment distribution plot: {str(e)}")
            return None

    def _plot_topic_sentiment(self, topic_sentiments: Dict, algorithm: str, output_dir: str) -> Optional[str]:
        try:
            data = []
            for topic, sentiment in topic_sentiments.items():
                data.append({
                    'Topic': topic,
                    'Average Score': sentiment['average_score'],
                    'Sentiment': sentiment['sentiment'],
                    'Sample Size': sentiment['sample_size']
                })

            if not data:
                return None

            df = pd.DataFrame(data)
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(
                data=df,
                x='Topic',
                y='Average Score',
                hue='Sentiment',
                palette={
                    'POSITIVE': self.palettes['sentiment']['positive'],
                    'NEUTRAL': self.palettes['sentiment']['neutral'],
                    'NEGATIVE': self.palettes['sentiment']['negative']
                },
                dodge=False
            )

            for i, p in enumerate(ax.patches):
                if i < len(df):
                    ax.annotate(
                        f"n={int(df['Sample Size'].iloc[i])}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10),
                        textcoords='offset points'
                    )

            plt.title(f'{algorithm.upper()} Topics - Average Sentiment Scores', pad=20)
            plt.xlabel('Topic')
            plt.ylabel('Average Sentiment Score (1-5)')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(1, 5)
            plt.axhline(3, color='gray', linestyle='--', alpha=0.5)
            plt.tight_layout()

            path = os.path.join(output_dir, f"{algorithm}_topic_average_sentiment.png")
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            return path
        except Exception as e:
            print(f"⚠️ Failed to generate average sentiment plot: {str(e)}")
            return None

    def _generate_combined_topic_sentiment_visualization(self, topic_data: Dict, sentiment_results: Dict,
                                                         algorithm: str):
        try:
            algo = algorithm.lower()
            topic_sentiments = self._analyze_topic_sentiments(sentiment_results, topic_data.get("topics", {}))
            if topic_sentiments:
                output_dir = os.path.join(self.output_path, f"{algo}_visualizations", "sentiment")
                os.makedirs(output_dir, exist_ok=True)

                filename = f"{algo}_topic_sentiment_summary.png"
                path = self._plot_topic_sentiment_summary(topic_sentiments, output_dir, filename)

                if path:
                    return path
        except Exception as e:
            print(f"⚠️ Failed to generate combined topic-sentiment visualization: {str(e)}")
        return None

    def _analyze_topic_sentiments(self, sentiment_results: Dict, topic_data: Dict) -> Dict:
        topic_sentiments = {}

        for topic_name, data in topic_data.items():
            keywords = set()
            for word in data['words'][:7]:
                doc = self.nlp(word.lower())
                keywords.update([token.lemma_ for token in doc])

            scores = []
            confused_count = total = 0

            for response in sentiment_results.values():
                sentences = []
                if 'sentences' in response:
                    sentences = response['sentences']
                elif isinstance(response, dict):
                    for q_data in response.values():
                        if isinstance(q_data, dict):
                            sentences.extend(q_data.get('sentences', []))

                for sentence in sentences:
                    if isinstance(sentence, dict):
                        text = sentence['text'].lower()
                        doc = self.nlp(text)
                        sentence_lemmas = set(token.lemma_ for token in doc)

                        if keywords & sentence_lemmas:
                            scores.append(sentence['sentiment']['score'])
                            if sentence['sentiment'].get('confusion', {}).get('status') == "CONFUSED":
                                confused_count += 1
                            total += 1

            if scores:
                topic_sentiments[topic_name] = {
                    "scores": scores,
                    "mean_score": np.mean(scores),
                    "positive_pct": len([s for s in scores if s >= 4]) / len(scores) * 100,
                    "confused_pct": confused_count / total * 100 if total > 0 else 0,
                    "count": total
                }
        return topic_sentiments

    def _plot_topic_sentiment_summary(self, topic_sentiments: Dict, output_dir: str, filename: str) -> Optional[str]:
        if not topic_sentiments:
            print("⚠️ No topic sentiment data to visualize")
            return None

        topics = list(topic_sentiments.keys())
        mean_scores = [data['mean_score'] for data in topic_sentiments.values()]
        positive_pcts = [data['positive_pct'] for data in topic_sentiments.values()]
        confused_pcts = [data['confused_pct'] for data in topic_sentiments.values()]
        counts = [data['count'] for data in topic_sentiments.values()]

        plt.figure(figsize=(14, 8))
        plt.suptitle('Topic Sentiment Analysis Summary', y=1.02, fontsize=16)

        # 1. Average Sentiment Plot
        plt.subplot(1, 2, 1)
        bars = plt.barh(
            topics,
            mean_scores,
            color=[self.palettes['sentiment']['positive'] if score >= 3
                   else self.palettes['sentiment']['negative'] for score in mean_scores],
            edgecolor='white'
        )

        for i, (score, count) in enumerate(zip(mean_scores, counts)):
            plt.text(
                max(score + 0.2, 3.2), i,
                f"n={count}",
                va='center',
                color='dimgrey'
            )
        plt.axvline(3, color='gray', linestyle='--', alpha=0.7)
        plt.title('Average Sentiment by Topic', pad=12)
        plt.xlabel('Mean Star Rating (1-5)')
        plt.xlim(1, 5)
        plt.grid(axis='x', alpha=0.2)

        # 2. Positive vs Confused Percentage Plot
        plt.subplot(1, 2, 2)
        x = np.arange(len(topics))
        width = 0.35

        pos_bars = plt.barh(
            x - width / 2, positive_pcts, width,
            label='Positive %',
            color=self.palettes['sentiment']['positive'],
            edgecolor='white'
        )
        conf_bars = plt.barh(
            x + width / 2, confused_pcts, width,
            label='Confused %',
            color=self.palettes['sentiment']['confused'],
            edgecolor='white'
        )

        for bar in pos_bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height() / 2,
                     f'{width:.1f}%', va='center', fontsize=9)
        for bar in conf_bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height() / 2,
                     f'{width:.1f}%', va='center', fontsize=9)

        plt.yticks(x, topics)
        plt.title('Positive vs Confused Responses', pad=12)
        plt.xlabel('Percentage')
        plt.legend(loc='lower right')
        plt.xlim(0, 100)
        plt.grid(axis='x', alpha=0.2)

        plt.tight_layout()
        path = os.path.join(output_dir, filename)
        plt.savefig(path, bbox_inches='tight', dpi=120)
        plt.close()
        return path

    def _generate_general_sentiment_visualizations(self, sentiment_results: Dict, algorithm: str) -> None:
        """Generate general sentiment visualizations in algorithm-specific folders"""
        algo = algorithm.lower()
        print(f"\n📈 Generating {algo.upper()} sentiment visualizations...")

        # Create algorithm-specific sentiment directory
        algo_sentiment_dir = os.path.join(self.output_path, f"{algo}_visualizations", "sentiment")
        os.makedirs(algo_sentiment_dir, exist_ok=True)

        visualizations = [
            ("histogram", self._plot_sentiment_histogram),
            ("pie", self._plot_confusion_pie),
            ("heatmap", self._plot_sentiment_confusion_heatmap)
        ]

        for name, viz_func in visualizations:
            path = viz_func(sentiment_results, algo_sentiment_dir)
            if path:
                print(f"  ✅ Generated {algo} sentiment {name}: {path}")

    def _plot_sentiment_histogram(self, sentiment_results: Dict, output_dir: str) -> Optional[str]:
        try:
            scores = []
            for response in sentiment_results.values():
                if 'overall_sentiment' in response:
                    scores.append(response['overall_sentiment'].get('score', 3))
                else:
                    for q_data in response.values():
                        if isinstance(q_data, dict) and 'overall_sentiment' in q_data:
                            scores.append(q_data['overall_sentiment'].get('score', 3))

            if not scores:
                return None

            plt.figure(figsize=(10, 5))
            ax = sns.histplot(
                scores, bins=5, binrange=(1, 6),
                discrete=True, stat='percent',
                color=self.palettes['sentiment']['neutral'],
                edgecolor='white'
            )

            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.1f}%',
                            (p.get_x() + p.get_width() / 2, height),
                            ha='center', va='bottom', xytext=(0, 3),
                            textcoords='offset points')

            plt.title('Sentiment Distribution (1-5 Stars)', pad=15)
            plt.xlabel('Star Rating')
            plt.ylabel('Percentage')
            plt.xticks(range(1, 6))
            plt.ylim(top=100)

            path = os.path.join(output_dir, "sentiment_histogram.png")
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            return path
        except Exception as e:
            print(f"⚠️ Failed to generate sentiment histogram: {str(e)}")
            return None

    def _plot_confusion_pie(self, sentiment_results: Dict, output_dir: str) -> Optional[str]:
        try:
            confused = clear = 0
            for response in sentiment_results.values():
                sentences = []
                if 'sentences' in response:
                    sentences = response['sentences']
                elif isinstance(response, dict):
                    for q_data in response.values():
                        if isinstance(q_data, dict):
                            sentences.extend(q_data.get('sentences', []))

                for sentence in sentences:
                    if isinstance(sentence, dict) and sentence.get('sentiment', {}).get('confusion', {}).get(
                            'status') == "CONFUSED":
                        confused += 1
                    else:
                        clear += 1

            if confused + clear == 0:
                return None

            plt.figure(figsize=(7, 7))
            plt.pie(
                [confused, clear],
                labels=['Confused', 'Clear'],
                colors=[self.palettes['sentiment']['confused'], self.palettes['sentiment']['clear']],
                autopct=lambda p: f'{p:.1f}%\n({int(p / 100 * (confused + clear))})',
                startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}
            )
            plt.title('Response Clarity', pad=20)

            path = os.path.join(output_dir, "confusion_pie.png")
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            return path
        except Exception as e:
            print(f"⚠️ Failed to generate confusion pie chart: {str(e)}")
            return None

    def _plot_sentiment_confusion_heatmap(self, sentiment_results: Dict, output_dir: str) -> Optional[str]:
        try:
            data = {
                '1 Star': {'Confused': 0, 'Clear': 0},
                '2 Stars': {'Confused': 0, 'Clear': 0},
                '3 Stars': {'Confused': 0, 'Clear': 0},
                '4 Stars': {'Confused': 0, 'Clear': 0},
                '5 Stars': {'Confused': 0, 'Clear': 0}
            }

            total = 0
            for response in sentiment_results.values():
                sentences = []
                if 'sentences' in response:
                    sentences = response['sentences']
                elif isinstance(response, dict):
                    for q_data in response.values():
                        if isinstance(q_data, dict):
                            sentences.extend(q_data.get('sentences', []))

                for sentence in sentences:
                    if isinstance(sentence, dict):
                        score = sentence.get('sentiment', {}).get('score', 3)
                        rating = f"{int(score)} Star{'s' if score > 1 else ''}"
                        status = 'Confused' if sentence.get('sentiment', {}).get('confusion', {}).get(
                            'status') == "CONFUSED" else 'Clear'
                        data[rating][status] += 1
                        total += 1

            if total == 0:
                return None

            df = pd.DataFrame(data).T
            df_percent = df.div(df.sum(axis=1), axis=0) * 100
            df_percent = df_percent.fillna(0)

            plt.figure(figsize=(10, 6))
            ax = sns.heatmap(
                df_percent,
                annot=True,
                fmt='.1f',
                cmap='YlOrRd',
                linewidths=0.5,
                linecolor='white',
                cbar_kws={'label': 'Percentage'},
                vmin=0,
                vmax=100
            )

            for t in ax.texts:
                t.set_text(t.get_text() + "%")

            plt.title('Sentiment vs Clarity Heatmap', pad=15)
            plt.xlabel('Clarity Status')
            plt.ylabel('Star Rating')

            path = os.path.join(output_dir, "sentiment_confusion_heatmap.png")
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            return path
        except Exception as e:
            print(f"⚠️ Failed to generate heatmap: {str(e)}")
            return None

