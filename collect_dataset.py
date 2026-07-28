import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

gesture_name = "Love" # Change this for each gesture
max_samples=400
sample_count=0

with open("dssetorg.csv", mode="a", newline="") as f:
    writer = csv.writer(f)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                base_x=landmarks[0]
                base_y=landmarks[1]
                normalized=[]
                for i in range(0,len(landmarks),2):
                    normalized.append(landmarks[i]-base_x)
                    normalized.append(landmarks[i+1]-base_y)
                row=normalized
                
                row.append(gesture_name)
                if sample_count<max_samples:
                    writer.writerow(row)
                    sample_count+=1
                if sample_count>=max_samples:
                    print("collected 400 samples")
                    break

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()