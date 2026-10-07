"""
Tests for EmotionAnalyzer in emotion_analyzer.py.

The fer package (and the TensorFlow model behind it) is replaced with a
stub, so these tests check the wrapper logic only and run without
TensorFlow installed.
"""
import importlib
import sys
import types

import numpy as np
import pytest

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
BOX = (100, 120, 80, 80)


class StubFER:
    """Stands in for fer.fer.FER and records how it is called."""

    def __init__(self, mtcnn=False):
        self.mtcnn = mtcnn
        self.results = []
        self.error = None
        self.calls = []

    def detect_emotions(self, img, face_rectangles=None):
        self.calls.append((img, face_rectangles))
        if self.error:
            raise self.error
        return self.results


@pytest.fixture
def analyzer(monkeypatch):
    fer_pkg = types.ModuleType("fer")
    fer_module = types.ModuleType("fer.fer")
    fer_module.FER = StubFER
    fer_pkg.fer = fer_module
    monkeypatch.setitem(sys.modules, "fer", fer_pkg)
    monkeypatch.setitem(sys.modules, "fer.fer", fer_module)
    monkeypatch.delitem(sys.modules, "emotion_analyzer", raising=False)

    module = importlib.import_module("emotion_analyzer")
    return module.EmotionAnalyzer()


def test_returns_top_emotion_and_score(analyzer):
    analyzer.detector.results = [
        {"box": BOX, "emotions": {"happy": 0.87, "sad": 0.05, "neutral": 0.08}}
    ]
    assert analyzer.analyze_face(FRAME, BOX) == ("happy", 0.87)


def test_passes_the_given_box_to_fer(analyzer):
    """The caller's box is reused, so FER does not detect faces again."""
    analyzer.analyze_face(FRAME, BOX)
    img, face_rectangles = analyzer.detector.calls[0]
    assert img is FRAME
    assert face_rectangles == [BOX]


def test_returns_none_when_fer_finds_nothing(analyzer):
    analyzer.detector.results = []
    assert analyzer.analyze_face(FRAME, BOX) == (None, 0.0)


def test_returns_none_when_fer_raises(analyzer):
    analyzer.detector.error = RuntimeError("model failure")
    assert analyzer.analyze_face(FRAME, BOX) == (None, 0.0)


def test_mtcnn_flag_is_forwarded(analyzer):
    module = sys.modules["emotion_analyzer"]
    assert analyzer.detector.mtcnn is False
    assert module.EmotionAnalyzer(mtcnn=True).detector.mtcnn is True
