import cv2
import mediapipe as mp
import pydirectinput


# Initialize video capture and MediaPipe Hands
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Define thresholds for hand movement detection
THRESHOLD_X = 0.05  # Adjust based on your needs

# Variables to track hand positions
previous_x = None

# Flags to track if actions have been performed
w_action = False
a_action = False

while True:
    _, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            
            # Get the position of the wrist (landmark 0)
            wrist = landmarks[0]
            x = wrist.x
            
            # Print the wrist coordinates for debugging
            print(f"Wrist Position: x={x}")

            # Check for hand movement to the right
            if previous_x is not None:
                if (x - previous_x) > THRESHOLD_X:  # Move right
                    if not a_action:
                        pydirectinput.keyDown('a')  # Hold down 'A'
                        a_action = True  # Mark action as performed
                else:
                    if a_action:
                        pydirectinput.keyUp('a')  # Release 'A'
                        a_action = False  # Reset action

                if (x - previous_x) < -THRESHOLD_X:  # Move left
                    if not w_action:
                        pydirectinput.keyDown('z')  # Hold down 'Z'
                        w_action = True  # Mark action as performed
                else:
                    if w_action:
                        pydirectinput.keyUp('z')  # Release 'Z'
                        w_action = False  # Reset action

            # Update previous positions
            previous_x = x

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
        break

cap.release()
cv2.destroyAllWindows()
