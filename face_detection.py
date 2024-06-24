import cv2
from deepface import DeepFace

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    try:
        # Analyze the frame with DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Extract the dominant emotion
        dominant_emotion = result[0]['dominant_emotion']
        
        # Draw the label on the frame
        cv2.putText(frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print("Error:", e)
    
    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('k'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
