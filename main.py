#!/usr/bin/env python3
"""
Sephardic Hebrew Bible Reader with Audio Processing
A comprehensive system for reading Hebrew Bible text with audio pronunciation matching
"""

import os
import json
import sqlite3
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import re
from datetime import datetime
import logging

# Web interface imports
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

# Audio processing imports
import soundfile as sf
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine

# Hebrew text processing
import unicodedata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HebrewVerse:
    """Data class for Hebrew Bible verses"""
    book: str
    chapter: int
    verse: int
    hebrew_text: str
    english_translation: str = ""
    audio_path: Optional[str] = None
    word_timings: Optional[List[Dict]] = None
    pronunciation_score: Optional[float] = None

@dataclass
class AudioSegment:
    """Data class for audio segments"""
    start_time: float
    end_time: float
    text: str
    features: Optional[np.ndarray] = None
    mfcc: Optional[np.ndarray] = None

class HebrewTextProcessor:
    """Handles Hebrew text processing and normalization"""

    def __init__(self):
        self.hebrew_letters = set("אבגדהוזחטיכלמנסעפצקרשת")
        self.vowels = set("ְֱֲֳִֵֶַָֹֻּׁׂ")
        self.cantillation = set("֑֖֛֢֣֤֥֦֧֪֚֭֮֒֓֔֕֗֘֙֜֝֞֟֠֡֨֩֫֬֯")

    def normalize_hebrew(self, text: str) -> str:
        """Normalize Hebrew text by removing or standardizing marks"""
        # Remove cantillation marks
        text = ''.join(char for char in text if char not in self.cantillation)

        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)

        return text

    def extract_words(self, text: str) -> List[str]:
        """Extract Hebrew words from text"""
        normalized = self.normalize_hebrew(text)
        words = re.findall(r'[א-ת]+(?:[ְֱֲֳִֵֶַָֹֻּׁׂ]*[א-ת]*)*', normalized)
        return [word.strip() for word in words if word.strip()]

    def remove_vowels(self, text: str) -> str:
        """Remove vowel marks from Hebrew text"""
        return ''.join(char for char in text if char not in self.vowels)

class AudioProcessor:
    """Handles audio processing and feature extraction using librosa"""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.n_mfcc = 13

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return None, None

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )
        return mfcc

    def detect_speech_segments(self, audio: np.ndarray, min_duration: float = 0.1) -> List[Tuple[float, float]]:
        """Detect speech segments in audio using energy-based detection"""
        # Compute RMS energy
        frame_length = 2048
        hop_length = 512

        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        # Normalize RMS
        rms = rms / np.max(rms)

        # Find speech segments using threshold
        threshold = 0.02
        speech_frames = rms > threshold

        # Convert frame indices to time
        times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=self.sample_rate,
            hop_length=hop_length
        )

        # Find continuous segments
        segments = []
        start = None

        for i, is_speech in enumerate(speech_frames):
            if is_speech and start is None:
                start = times[i]
            elif not is_speech and start is not None:
                end = times[i]
                if end - start >= min_duration:
                    segments.append((start, end))
                start = None

        # Handle case where speech continues to end
        if start is not None:
            segments.append((start, times[-1]))

        return segments

    def align_text_to_audio(self, audio: np.ndarray, words: List[str]) -> List[Dict]:
        """Align Hebrew words to audio segments"""
        segments = self.detect_speech_segments(audio)

        if len(segments) == 0:
            return []

        # Simple alignment: distribute words evenly across segments
        word_timings = []

        if len(words) <= len(segments):
            # More segments than words
            for i, word in enumerate(words):
                if i < len(segments):
                    start, end = segments[i]
                    word_timings.append({
                        'word': word,
                        'start': start,
                        'end': end,
                        'segment_index': i
                    })
        else:
            # More words than segments - distribute evenly
            total_duration = sum(end - start for start, end in segments)
            avg_word_duration = total_duration / len(words)

            current_time = segments[0][0]
            segment_idx = 0

            for word in words:
                start = current_time
                end = min(current_time + avg_word_duration, segments[-1][1])

                # Ensure we don't exceed current segment
                while segment_idx < len(segments) - 1 and end > segments[segment_idx][1]:
                    segment_idx += 1
                    if segment_idx < len(segments):
                        current_time = segments[segment_idx][0]
                        start = current_time
                        end = min(current_time + avg_word_duration, segments[segment_idx][1])

                word_timings.append({
                    'word': word,
                    'start': start,
                    'end': end,
                    'segment_index': segment_idx
                })

                current_time = end

        return word_timings

    def compare_pronunciations(self, audio1: np.ndarray, audio2: np.ndarray) -> float:
        """Compare two audio pronunciations using MFCC features"""
        mfcc1 = self.extract_mfcc(audio1)
        mfcc2 = self.extract_mfcc(audio2)

        # Compute mean MFCC vectors
        mean_mfcc1 = np.mean(mfcc1, axis=1)
        mean_mfcc2 = np.mean(mfcc2, axis=1)

        # Compute cosine similarity
        similarity = 1 - cosine(mean_mfcc1, mean_mfcc2)

        return max(0, similarity)  # Ensure non-negative

class HebrewBibleDatabase:
    """Handles database operations for Hebrew Bible text and audio data"""

    def __init__(self, db_path: str = "hebrew_bible.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create verses table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS verses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book TEXT NOT NULL,
                    chapter INTEGER NOT NULL,
                    verse INTEGER NOT NULL,
                    hebrew_text TEXT NOT NULL,
                    english_translation TEXT,
                    audio_path TEXT,
                    word_timings TEXT,
                    pronunciation_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(book, chapter, verse)
                )
            ''')

            # Create audio_segments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audio_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    verse_id INTEGER,
                    start_time REAL,
                    end_time REAL,
                    text TEXT,
                    mfcc_features TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (verse_id) REFERENCES verses (id)
                )
            ''')

            conn.commit()

    def add_verse(self, verse: HebrewVerse) -> int:
        """Add a verse to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            word_timings_json = json.dumps(verse.word_timings) if verse.word_timings else None

            cursor.execute('''
                INSERT OR REPLACE INTO verses
                (book, chapter, verse, hebrew_text, english_translation, audio_path, word_timings, pronunciation_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                verse.book, verse.chapter, verse.verse, verse.hebrew_text,
                verse.english_translation, verse.audio_path, word_timings_json, verse.pronunciation_score
            ))

            return cursor.lastrowid

    def get_verse(self, book: str, chapter: int, verse: int) -> Optional[HebrewVerse]:
        """Get a specific verse from the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT book, chapter, verse, hebrew_text, english_translation, audio_path, word_timings, pronunciation_score
                FROM verses WHERE book = ? AND chapter = ? AND verse = ?
            ''', (book, chapter, verse))

            row = cursor.fetchone()
            if row:
                word_timings = json.loads(row[6]) if row[6] else None
                return HebrewVerse(
                    book=row[0], chapter=row[1], verse=row[2], hebrew_text=row[3],
                    english_translation=row[4], audio_path=row[5], word_timings=word_timings,
                    pronunciation_score=row[7]
                )

        return None

    def get_chapter(self, book: str, chapter: int) -> List[HebrewVerse]:
        """Get all verses from a specific chapter"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT book, chapter, verse, hebrew_text, english_translation, audio_path, word_timings, pronunciation_score
                FROM verses WHERE book = ? AND chapter = ? ORDER BY verse
            ''', (book, chapter))

            verses = []
            for row in cursor.fetchall():
                word_timings = json.loads(row[6]) if row[6] else None
                verses.append(HebrewVerse(
                    book=row[0], chapter=row[1], verse=row[2], hebrew_text=row[3],
                    english_translation=row[4], audio_path=row[5], word_timings=word_timings,
                    pronunciation_score=row[7]
                ))

            return verses

class HebrewBibleReader:
    """Main class for the Hebrew Bible reader application"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.text_processor = HebrewTextProcessor()
        self.audio_processor = AudioProcessor()
        self.database = HebrewBibleDatabase()

        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/api/books')
        def get_books():
            """Get list of available books"""
            # This should be populated from your database
            books = [
                "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
                "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
                "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
                "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Proverbs",
                "Ecclesiastes", "Song of Songs", "Isaiah", "Jeremiah", "Lamentations",
                "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah",
                "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai",
                "Zechariah", "Malachi"
            ]
            return jsonify(books)

        @self.app.route('/api/verse/<book>/<int:chapter>/<int:verse>')
        def get_verse_api(book, chapter, verse):
            """Get a specific verse"""
            verse_obj = self.database.get_verse(book, chapter, verse)
            if verse_obj:
                return jsonify(asdict(verse_obj))
            return jsonify({"error": "Verse not found"}), 404

        @self.app.route('/api/chapter/<book>/<int:chapter>')
        def get_chapter_api(book, chapter):
            """Get all verses from a chapter"""
            verses = self.database.get_chapter(book, chapter)
            return jsonify([asdict(v) for v in verses])

        @self.app.route('/api/process-audio', methods=['POST'])
        def process_audio():
            """Process uploaded audio file"""
            if 'audio' not in request.files:
                return jsonify({"error": "No audio file provided"}), 400

            audio_file = request.files['audio']
            hebrew_text = request.form.get('hebrew_text', '')

            if not hebrew_text:
                return jsonify({"error": "Hebrew text is required"}), 400

            # Save uploaded file
            audio_path = self.data_dir / "temp" / audio_file.filename
            audio_path.parent.mkdir(exist_ok=True)
            audio_file.save(audio_path)

            # Process audio
            result = self.process_audio_file(str(audio_path), hebrew_text)

            # Clean up temp file
            audio_path.unlink()

            return jsonify(result)

    def process_audio_file(self, audio_path: str, hebrew_text: str) -> Dict:
        """Process an audio file and align it with Hebrew text"""
        # Load audio
        audio, sr = self.audio_processor.load_audio(audio_path)
        if audio is None:
            return {"error": "Could not load audio file"}

        # Extract Hebrew words
        words = self.text_processor.extract_words(hebrew_text)

        # Align text to audio
        word_timings = self.audio_processor.align_text_to_audio(audio, words)

        # Extract MFCC features for each word segment
        for timing in word_timings:
            start_sample = int(timing['start'] * sr)
            end_sample = int(timing['end'] * sr)
            word_audio = audio[start_sample:end_sample]

            if len(word_audio) > 0:
                mfcc = self.audio_processor.extract_mfcc(word_audio)
                timing['mfcc_features'] = mfcc.tolist()

        return {
            "word_timings": word_timings,
            "total_duration": len(audio) / sr,
            "num_words": len(words),
            "words": words
        }

    def load_sample_data(self):
        """Load sample Hebrew Bible data"""
        # Sample verse from Genesis 1:1
        sample_verse = HebrewVerse(
            book="Genesis",
            chapter=1,
            verse=1,
            hebrew_text="בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ",
            english_translation="In the beginning God created the heaven and the earth."
        )

        self.database.add_verse(sample_verse)
        logger.info("Sample data loaded")

    def run(self, host='127.0.0.1', port=5000, debug=True):
        """Run the Flask application"""
        self.load_sample_data()
        logger.info(f"Starting Hebrew Bible Reader on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Main entry point"""
    reader = HebrewBibleReader()
    reader.run()

if __name__ == "__main__":
    main()