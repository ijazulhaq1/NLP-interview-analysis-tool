import os
import nltk
import json
from typing import Dict, List
from json_handler import JsonHandler
from preprocessing import Preprocessor
from topic_modeling import TopicModeler
from sentiment_analysis import SentimentAnalyzer
from visualization import Visualizer
import logging
from datetime import datetime
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TerminalInterface:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.preprocessor = Preprocessor()
        self.analyzer = SentimentAnalyzer()
        self.visualizer = Visualizer()
        self.stopwords = nltk.corpus.stopwords.words("spanish")
        self.file_path = './data/output'
        self.current_topic_algorithm = None

    def _select_questions(self) -> Dict:
        """Display all questions and let user select which to analyze"""
        input_file = "./data/input/interviews.json"
        if not os.path.exists(input_file):
            logger.error(f"Input file not found at {input_file}")
            return {}

        raw_data = JsonHandler.read_json(input_file)
        if not raw_data:
            logger.error("No data found in input file")
            return {}

        # Collect all unique questions across interviews
        all_questions = set()
        for interview in raw_data.values():
            all_questions.update(interview.keys())

        # Sort questions numerically
        sorted_questions = sorted(all_questions, key=lambda x: [int(i) for i in x.split('.') if i.isdigit()])

        print("\n📋 Available Questions:")
        for i, question in enumerate(sorted_questions, 1):
            print(f"{i}. {question}")

        print("\nSelect questions to analyze (comma-separated numbers, or 'all'):")
        selection = input("Your selection: ").strip().lower()

        selected_questions = []
        if selection == 'all':
            selected_questions = sorted_questions
            print("✅ Selected all questions")
        else:
            try:
                selected_indices = [int(i.strip()) for i in selection.split(',')]
                selected_questions = [sorted_questions[i - 1] for i in selected_indices]
                print(f"✅ Selected questions: {', '.join(selected_questions)}")
            except Exception as e:
                logger.error(f"Invalid selection: {str(e)}")
                return {}

        # Filter the data to only include selected questions
        filtered_data = {}
        for interview_id, questions in raw_data.items():
            filtered_questions = {q: text for q, text in questions.items() if q in selected_questions}
            if filtered_questions:
                filtered_data[interview_id] = filtered_questions

        return filtered_data

    def _create_processed_versions(self, source_data: Dict) -> bool:
        """Create processed versions of the data"""
        full_processed = {}
        sentiment_ready = {}
        word_blacklist = self.stopwords + ["día", "vez", "25", "año", "pues", "entonces"]

        for interview, questions in source_data.items():
            full_processed[interview] = {}
            sentiment_ready[interview] = {}

            for question, text in questions.items():
                translated = self.preprocessor.translate_to_catalan(text)

                # Version for topic modeling
                processed = self.preprocessor.full_preprocess(translated)
                full_processed[interview][question] = processed

                # Version for sentiment analysis
                sentences = self.preprocessor.split_sentences(translated, "es")
                sentiment_ready[interview][question] = sentences

        # Save files
        JsonHandler.create_json(full_processed, "fully_processed_ca.json", self.file_path)
        JsonHandler.create_json(sentiment_ready, "sentiment_input_ca.json", self.file_path)
        JsonHandler.create_topic_modeling_input(full_processed, self.file_path)

        logger.info("Created all processed versions")
        return True

    def _analyze_topics(self) -> bool:
        """Run topic modeling using the dedicated input file"""
        input_file = os.path.join(self.file_path, "topic_modeling_input.json")

        if not os.path.exists(input_file):
            logger.warning("Topic modeling input not found. Creating processed versions...")
            if not self._create_processed_versions():
                return False

        try:
            documents = JsonHandler.load_topic_modeling_input(input_file)
            if not documents:
                logger.error("No valid documents found in topic modeling input")
                return False

            logger.info(f"Loaded {len(documents)} documents for topic modeling")
            if len(documents) < 10:
                logger.error(f"Insufficient documents ({len(documents)}). Need ≥10.")
                return False

            # Get algorithm choice from user
            print("\nAvailable topic modeling algorithms:")
            print("1. BERTopic (default)")
            print("2. LDA")
            print("3. NMF")
            print("4. SVD")
            choice = input("Select algorithm (1-4): ").strip() or "1"

            algorithm_map = {
                "1": "bertopic",
                "2": "lda",
                "3": "nmf",
                "4": "svd"
            }
            algorithm = algorithm_map.get(choice, "bertopic")
            self.current_topic_algorithm = algorithm

            # Run topic modeling
            topics = TopicModeler.extract_topics(
                documents,
                algorithm=algorithm,
                n_topics="auto" if algorithm == "bertopic" else 10,
                n_words=10,
                stopwords=self.stopwords
            )

            # Save results with algorithm-specific filename
            output_filename = f"topic_results_{algorithm}.json"  # This line already uses the algorithm name
            JsonHandler.create_json(
                {
                    "topics": topics.get("topics", {}),
                    "metadata": {
                        "source": "topic_modeling_input.json",
                        "document_count": len(documents),
                        "algorithm": algorithm,  # This includes the algorithm in metadata
                        "n_topics_requested": "auto" if algorithm == "bertopic" else 10,
                        "random_state": 42,
                        "generated_at": datetime.now().isoformat()
                    },
                    "diagnostics": topics.get("diagnostics", {})
                },
                output_filename,
                self.file_path
            )

            logger.info(f"Topic analysis using {algorithm.upper()} completed successfully")
            return True  # ✅ FIXED: Added missing return statement here

        except Exception as e:
            logger.error(f"Topic analysis failed: {str(e)}")
            return False

    def _analyze_sentiment(self) -> bool:
        """Run sentiment analysis using sentiment_input_es.json"""
        input_file = os.path.join(self.file_path, "sentiment_input_ca.json")
        if not os.path.exists(input_file):
            logger.warning("Sentiment input not found. Creating processed versions...")
            if not self._create_processed_versions():
                return False

        try:
            data = JsonHandler.read_json(input_file)
            if not data:
                logger.error("No data available for sentiment analysis")
                return False

            results = {}
            total_sentences = 0

            # Analyze each interview and question
            for interview_id, questions in tqdm(data.items(), desc="Analyzing interviews"):
                results[interview_id] = {}
                for question_id, sentences in questions.items():
                    sentence_results = []
                    for sentence in sentences:
                        analysis = self.analyzer.analyze(sentence)
                        sentence_results.append({
                            "text": sentence,
                            "sentiment": analysis
                        })
                        total_sentences += 1

                    # Calculate overall sentiment for this question
                    if sentence_results:
                        avg_score = sum(s['sentiment']['score'] for s in sentence_results) / len(sentence_results)
                        confusion_count = sum(1 for s in sentence_results
                                          if s['sentiment']['confusion'] and
                                          s['sentiment']['confusion']['status'] == "CONFUSED")

                        results[interview_id][question_id] = {
                            "sentences": sentence_results,
                            "overall_sentiment": {
                                "score": avg_score,
                                "confusion_percentage": (confusion_count / len(sentence_results)) * 100
                            }
                        }

            # Save results
            JsonHandler.create_json(
                results,
                "sentiment_results_ca.json",
                self.file_path
            )

            logger.info(f"Analyzed {total_sentences} sentences across {len(data)} interviews")
            return True

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return False

    def _visualize_results(self):
        """Generate visualizations with algorithm-specific organization"""
        try:
            # Check for sentiment results
            sentiment_file = os.path.join(self.file_path, "sentiment_results_ca.json")
            if not os.path.exists(sentiment_file):
                logger.error("No sentiment analysis results found. Run sentiment analysis first.")
                return

            sentiment_results = JsonHandler.read_json(sentiment_file)
            if not sentiment_results:
                logger.error("Could not load sentiment results")
                return

            # Find all topic result files
            topic_files = []
            for filename in os.listdir(self.file_path):
                if filename.startswith("topic_results_") and filename.endswith(".json"):
                    topic_files.append(os.path.join(self.file_path, filename))

            if not topic_files:
                logger.error("No topic analysis results found. Run topic analysis first.")
                return

            # Process each topic result file
            for topic_file in topic_files:
                try:
                    topic_results = JsonHandler.read_json(topic_file)
                    if not topic_results:
                        continue

                    # Get algorithm from filename
                    algo = os.path.basename(topic_file).split('_')[2].split('.')[0].lower()

                    # Generate visualizations
                    self.visualizer.generate_all_visualizations(
                        topic_data=topic_results,
                        sentiment_results=sentiment_results
                    )

                except Exception as e:
                    logger.error(f"Failed to process {topic_file}: {str(e)}")

            logger.info("Visualizations generated successfully")
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")

    def run(self):
        """Main interface menu"""
        while True:
            print("\n📋 Main Menu:")
            print("1. Select questions and create processed versions")
            print("2. Analyze topics")
            print("3. Analyze sentiment")
            print("4. Visualize results")
            print("0. Exit")

            try:
                choice = input("Enter choice (0-4): ").strip()
                if choice == "1":
                    selected_data = self._select_questions()
                    if selected_data:
                        self._create_processed_versions(selected_data)
                elif choice == "2":
                    self._analyze_topics()
                elif choice == "3":
                    self._analyze_sentiment()
                elif choice == "4":
                    self._visualize_results()
                elif choice == "0":
                    print("\n👋 Exiting program.")
                    break
                else:
                    print("⚠️ Invalid choice. Please try again.")
            except Exception as e:
                logger.error(f"Menu error: {str(e)}")