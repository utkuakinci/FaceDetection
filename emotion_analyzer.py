from fer.fer import FER

class EmotionAnalyzer:
    def __init__(self, mtcnn: bool = False):
        # Initialize the detector once to avoid reloading models on every frame
        self.detector = FER(mtcnn=mtcnn)

    def analyze_face(self, frame, box):
        """
        Returns the top emotion for the face at box (x, y, w, h) in frame.

        The box is handed to FER as-is, so FER does not run its own face
        detection on top of the one already done by the caller.
        """
        try:
            results = self.detector.detect_emotions(frame, face_rectangles=[box])
            if not results:
                return None, 0.0
            emotions = results[0]["emotions"]
            emotion = max(emotions, key=emotions.get)
            return emotion, emotions[emotion]
        except Exception:
            return None, 0.0