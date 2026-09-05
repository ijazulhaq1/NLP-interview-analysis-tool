import os
import re
import string
import nltk
import spacy
from nltk.corpus import stopwords
from typing import List, Tuple
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException


class Preprocessor:
    def __init__(self, extra_stopwords: List[str] = None):
        nltk.download('stopwords', quiet=True)

        try:
            # Load language models
            self.nlp_es = spacy.load("es_core_news_sm")
            self.nlp_ca = spacy.load("ca_core_news_sm")
            self.nlp_en = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download es_core_news_sm")
            os.system("python -m spacy download ca_core_news_sm")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp_es = spacy.load("es_core_news_sm")
            self.nlp_ca = spacy.load("ca_core_news_sm")
            self.nlp_en = spacy.load("en_core_web_sm")

        # Load stopwords
        self.stopwords_es = set(stopwords.words("spanish"))
        self.stopwords_ca = set(stopwords.words("catalan"))
        self.stopwords_en = set(stopwords.words("english"))

        if extra_stopwords:
            self.stopwords_es.update(word.lower() for word in extra_stopwords)
            self.stopwords_ca.update(word.lower() for word in extra_stopwords)

        self.essential_verbs = {'ser', 'estar', 'tener', 'haver', 'fer', 'hacer'}

    def detect_language(self, text: str) -> str:
        """
        Detect language using langdetect (more stable than GoogleTranslator).
        Returns 'ca', 'es', or 'en'.
        """
        try:
            lang = detect(text[:500])
            if lang in ['ca', 'es', 'en']:
                print(f"[LangDetect] Detected language: {lang}")
                return lang
            return 'ca'  # Default fallback
        except LangDetectException:
            print("[LangDetect] Detection failed, defaulting to Catalan.")
            return 'ca'

    def translate_to_catalan(self, text: str) -> str:
        """
        Translate text to Catalan if it's not already Catalan.
        """
        try:
            lang = self.detect_language(text)
            if lang == 'ca':
                return text

            max_chunk_size = 4000
            chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

            translated_chunks = []
            for chunk in chunks:
                translated = GoogleTranslator(source='auto', target='ca').translate(chunk)
                translated_chunks.append(translated)

            return ' '.join(translated_chunks)

        except Exception as e:
            print(f"[Translate] Translation failed: {e}")
            return text

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        # Remove timestamps
        text = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]", "", text)
        # Remove special characters but keep accented letters
        text = re.sub(r"[^\w\sáéíóúÁÉÍÓÚñÑ.,!?¡¿]", " ", text)
        # Remove standalone numbers
        text = ' '.join([word for word in text.split() if not word.isdigit()])
        return " ".join(text.split())

    def split_sentences(self, text: str, lang: str = "ca", chunk_size: int = 150) -> List[str]:
        if not text.strip():
            return []

        nlp = {
            'es': self.nlp_es,
            'ca': self.nlp_ca,
            'en': self.nlp_en
        }.get(lang, self.nlp_ca)

        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if not sentences or (len(sentences) == 1 and len(sentences[0]) > chunk_size):
            sentences = []
            words = text.split()
            current_chunk = []
            current_length = 0

            for word in words:
                if current_length + len(word) > chunk_size and current_chunk:
                    sentences.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(word)
                current_length += len(word) + 1

            if current_chunk:
                sentences.append(' '.join(current_chunk))

        return sentences

    def preprocess_with_punct(self, text: str, lang: str = "ca") -> str:
        text = self.clean_text(text)
        text = self.lemmatize_with_punct(text, lang)
        return text

    def lemmatize_with_punct(self, text: str, lang: str = "ca") -> str:
        nlp = {
            'es': self.nlp_es,
            'ca': self.nlp_ca,
            'en': self.nlp_en
        }.get(lang, self.nlp_ca)

        stopwords_set = {
            'es': self.stopwords_es,
            'ca': self.stopwords_ca,
            'en': self.stopwords_en
        }.get(lang, self.stopwords_ca)

        doc = nlp(text)
        tokens = []

        for token in doc:
            if token.is_punct:
                tokens.append(token.text)
            elif not token.is_space:
                lemma = token.lemma_.lower()
                if (lemma not in stopwords_set or lemma in self.essential_verbs):
                    tokens.append(lemma)

        return " ".join(tokens)

    def remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation + '¡¿'))

    def process_mixed_language(self, text: str) -> Tuple[str, str]:
        lang = self.detect_language(text)

        if lang == 'es':
            translated = self.translate_to_catalan(text)
            processed = self.full_preprocess(translated, 'ca')
            return processed, 'ca'
        else:
            processed = self.full_preprocess(text, lang)
            return processed, lang

    def full_preprocess(self, text: str, lang: str = "ca") -> str:
        """Add additional cleaning"""
        with_punct = self.preprocess_with_punct(text, lang)
        sentences = self.split_sentences(with_punct, lang)
        processed_sentences = []
        for sent in sentences:
            # Remove punctuation and clean each sentence
            sent = self.remove_punctuation(sent)
            sent = ' '.join([word for word in sent.split() if not word.isdigit()])
            if sent:
                processed_sentences.append(sent)
        return " ".join(processed_sentences)
