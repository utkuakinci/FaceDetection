from fer import FER

class EmotionAnalyzer:
    def __init__(self, mtcnn: bool = False):
        # Initialize the detector once to avoid reloading models on every frame
        self.detector = FER(mtcnn=mtcnn)

    def analyze_face(self, face_img):
        """
        Analyzes a cropped face region or full frame and returns the top emotion.
        """
        try:
            emotion, score = self.detector.top_emotion(face_img)
            return emotion, score
        except Exception:
            return None, 0.0