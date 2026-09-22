import cv2
from emotion_analyzer import EmotionAnalyzer

def main():
    # Start the camera
    cap = cv2.VideoCapture(0)
    # Install the Haar Cascade face classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Instantiate the analyzer outside the frame loop
    analyzer = EmotionAnalyzer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Gri tonlara çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Yüzleri tespit et
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Yüzlerin etrafına dikdörtgen çiz
        for (x, y, w, h) in faces:
            # Crop the detected face ROI (Region of Interest)
            face_roi = frame[y:y+h, x:x+w]
            
            # Get emotion prediction
            emotion, score = analyzer.analyze_face(face_roi)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if emotion:
                label = f"{emotion} ({int(score * 100)}%)"
                cv2.putText(
                    frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2
                )

        cv2.imshow("Face & Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()