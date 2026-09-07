import os
import re
import string
import nltk
import spacy
from nltk.corpus import stopwords
from typing import List, Tuple
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException, DetectorFactory

# langdetect's detection algorithm is probabilistic (it samples n-gram
# features); without a fixed seed, the exact language guessed for a short
# or ambiguous string can vary between runs/processes, which then cascades
# into which translation branch is taken, whether translation happens at
# all, and the final sentence representation. Setting this once at import
# time, before any Detector is constructed, makes every langdetect call in
# this process (here and in preprocess_v2.py, which imports this module)
# deterministic across runs. Values already using detect()/detect_langs()
# on identical text keep behaving exactly the same as before for anything
# unambiguous, but formerly run-to-run-unstable detections on ambiguous
# text now settle to a single stable answer.
DetectorFactory.seed = 0


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
        Used for the topic-modelling path only (unchanged from V8).
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

    def translate_to_english(self, text: str) -> str:
        """
        Translate text to English if it's not already English.

        Used for the sentiment/confusion path (Step 2 fix): the current
        classifiers are nlptown/bert-base-multilingual-uncased-sentiment
        (fine-tuned on en/nl/de/fr/es/it — not Catalan) and
        facebook/bart-large-mnli (English-only). Neither model has any
        training exposure to Catalan, so text must not be pivoted through
        Catalan before reaching them. English is the one language both
        models are confidently in-distribution for, so it is used here as
        a dedicated pivot for classification purposes only — this is
        independent of translate_to_catalan(), which remains the pivot
        for topic modelling and is unchanged.
        """
        try:
            lang = self.detect_language(text)
            if lang == 'en':
                return text

            max_chunk_size = 4000
            chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

            translated_chunks = []
            for chunk in chunks:
                translated = GoogleTranslator(source='auto', target='en').translate(chunk)
                translated_chunks.append(translated)

            return ' '.join(translated_chunks)

        except Exception as e:
            print(f"[Translate] English translation failed: {e}")
            return text

    def translate_to_spanish(self, text: str) -> str:
        """
        Translate text to Spanish if it's not already Spanish.

        Used for the sentiment path (Step 2 fix, revised design): the
        5-star sentiment classifier (nlptown/bert-base-multilingual-
        uncased-sentiment) is fine-tuned on en/nl/de/fr/es/it — it has
        real training exposure to Spanish, unlike Catalan. Spanish is
        also the language closest to how most interviews were actually
        conducted, so it is used here as the dedicated pivot for the
        sentiment classifier only. This is independent of
        translate_to_catalan() (topic-modelling pivot) and
        translate_to_english() (confusion/BART pivot).
        """
        try:
            lang = self.detect_language(text)
            if lang == 'es':
                return text

            max_chunk_size = 4000
            chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

            translated_chunks = []
            for chunk in chunks:
                translated = GoogleTranslator(source='auto', target='es').translate(chunk)
                translated_chunks.append(translated)

            return ' '.join(translated_chunks)

        except Exception as e:
            print(f"[Translate] Spanish translation failed: {e}")
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

    def clean_text_unicode_safe(self, text: str) -> str:
        """Unicode/Catalan-safe cleaning for the Step 2 v2 topic-cleaning
        path (topic_text_ca_clean).

        clean_text() above whitelists an explicitly Spanish set of
        accented letters (áéíóúÁÉÍÓÚñÑ). \\w is already Unicode-aware in
        Python 3 str patterns, so à/è/ò/ç/ï/ü were never actually at risk
        from that regex — but clean_text()'s whitelist does not include
        the apostrophe (') or the interpunct (·), so both get replaced
        with a space. That silently corrupts real Catalan orthography:
        "col·laboració" -> "col laboració", which spaCy's ca tokenizer
        then splits into two separate, meaningless lexical tokens ("col",
        "laboració") instead of keeping the one real word intact —
        verified empirically against the ca_core_news_sm tokenizer. This
        matters because topic_text_ca_clean feeds lexical topic models
        (NMF/LDA/SVD/c-TF-IDF) where a word is its exact token.

        clean_text() itself is left untouched: it still feeds
        fully_processed_ca.json / topic_modeling_input.json via
        terminal.py's frozen Step-1 pipeline (through full_preprocess()),
        and that path must keep producing exactly what it already
        produces until an explicitly-approved later step switches it
        over to the v2 outputs.
        """
        text = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]", "", text)
        # Keep: any Unicode word character, whitespace, sentence
        # punctuation, and the two Catalan-orthography-bearing marks
        # (apostrophe incl. curly '’', and interpunct '·'). Hyphen is
        # also kept (conservative default — retains compound words and
        # this transcript's leading "-" speaker-turn marker).
        text = re.sub(r"[^\w\s.,!?¡¿'’·-]", " ", text, flags=re.UNICODE)
        text = ' '.join([word for word in text.split() if not word.isdigit()])
        return " ".join(text.split())

    def full_preprocess_v2(self, text: str, lang: str = "ca") -> str:
        """Topic-text cleaning for Step 2 v2 (topic_text_ca_clean).

        clean -> lemmatize/stopword-removal -> drop standalone digits and
        standalone punctuation-only tokens. Deliberately does NOT call
        split_sentences() at any point — full_preprocess() above chunks
        internally via the legacy 150-char fallback (harmless there only
        because the chunks are immediately rejoined into one string), but
        this path is built to be fully independent of that legacy
        behavior rather than incidentally unaffected by it.

        Only standalone punctuation tokens (a lone '.', ',', etc.) are
        dropped — apostrophes/interpunct embedded inside a real word
        token (e.g. "col·laboració", "l'") are never touched, since they
        only ever appear word-internally once clean_text_unicode_safe has
        run, never as their own token.
        """
        cleaned = self.clean_text_unicode_safe(text)
        lemmatized = self.lemmatize_with_punct(cleaned, lang)
        pure_punct = re.compile(r"^[.,!?¡¿\-]+$")
        tokens = [
            w for w in lemmatized.split()
            if not w.isdigit() and not pure_punct.match(w)
        ]
        return " ".join(tokens)

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

    def split_sentences_strict(self, text: str, lang: str = "ca") -> List[str]:
        """Sentence splitting using spaCy sentence boundaries ONLY.

        Unlike split_sentences() above, this never re-chunks a single long
        spaCy sentence into arbitrary ~150-character fragments — a genuine
        sentence, however long, stays one sentence. Any model token-length
        handling belongs at the classifier/model-input stage, not here:
        changing the statistical unit from "sentence" to "arbitrary
        character chunk" is exactly the kind of thing a reviewer asking
        about sentence-level classification and conversational meaning
        would object to.

        This is a separate method (not a change to split_sentences()
        itself) because terminal.py's Step-1-approved pipeline still calls
        split_sentences() directly for sentiment_input_ca.json; that path
        is frozen and must keep producing exactly what it already
        produces. split_sentences_strict() is for preprocess_v2.py only.
        """
        if not text.strip():
            return []

        nlp = {
            'es': self.nlp_es,
            'ca': self.nlp_ca,
            'en': self.nlp_en
        }.get(lang, self.nlp_ca)

        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

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
