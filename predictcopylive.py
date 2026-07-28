import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
pred_queue=deque(maxlen=10)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("camera not working")
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
      for hand_landmarks, handedness in zip(result.multi_hand_landmarks,result.multi_handedness):
          hand_label=handedness.classification[0].label
          if hand_label=="Right":
            # Draw landmarks
             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
             landmarks= []
             for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
             base_x=landmarks[0]
             base_y=landmarks[1]
             normalized=[]
             for i in range(0,len(landmarks),2):
                normalized.append(landmarks[i]-base_x)
                normalized.append(landmarks[i+1]-base_y)

             data = np.array(normalized).reshape(1, -1)

             probs=model.predict_proba(data)[0]
             confidence=max(probs)
             pred_index=probs.argmax()
             prediction=label_encoder.inverse_transform([pred_index])[0]
             if confidence<0.6:
                label="Unknown"
             else:
                label=prediction
            


            # display
             text=f"{label} ({confidence:.2f})"
             cv2.putText(frame, text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()