"""
Tests for the Haar Cascade face detection logic used in main.py.

main.py runs its detection loop at module level (it opens the webcam
on import), so it can't be imported directly in a test environment.
These tests instead exercise the same OpenCV Haar Cascade pipeline
in isolation, on static images, so they run headlessly (no webcam
needed) and deterministically.

Run with:
    pip install -r tests/requirements-test.txt
    pytest tests/
"""
import os

import cv2
import numpy as np
import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SAMPLE_FACE_IMAGE = os.path.join(DATA_DIR, "sample_face.jpg")


@pytest.fixture(scope="module")
def face_cascade():
    """Load the same Haar Cascade classifier used in main.py."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    assert not cascade.empty(), f"Failed to load cascade from {cascade_path}"
    return cascade


def detect_faces(cascade, image):
    """Mirrors the detection call in main.py."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )


def test_cascade_file_loads(face_cascade):
    """The bundled Haar Cascade XML should load without error."""
    assert face_cascade is not None
    assert not face_cascade.empty()


def test_no_false_positive_on_blank_image(face_cascade):
    """A blank black frame should not be detected as containing a face."""
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = detect_faces(face_cascade, blank_frame)
    assert len(faces) == 0


def test_no_false_positive_on_random_noise(face_cascade):
    """Random noise should not reliably trigger a face detection."""
    rng = np.random.default_rng(seed=42)
    noise_frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    faces = detect_faces(face_cascade, noise_frame)
    # Noise can occasionally produce a stray box; there should never be many.
    assert len(faces) <= 1


@pytest.mark.skipif(
    not os.path.exists(SAMPLE_FACE_IMAGE),
    reason=(
        "No sample face image found. Add one at "
        "tests/data/sample_face.jpg to enable this test."
    ),
)
def test_detects_face_in_sample_image(face_cascade):
    """With a real face image present, the cascade should find at least one face."""
    image = cv2.imread(SAMPLE_FACE_IMAGE)
    assert image is not None, f"Could not read {SAMPLE_FACE_IMAGE}"
    faces = detect_faces(face_cascade, image)
    assert len(faces) >= 1

    # Each detection should be a valid (x, y, w, h) box within the image bounds.
    height, width = image.shape[:2]
    for (x, y, w, h) in faces:
        assert 0 <= x < width and 0 <= y < height
        assert w > 0 and h > 0
        assert x + w <= width and y + h <= height
