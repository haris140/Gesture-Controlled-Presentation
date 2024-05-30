import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Initialize MediaPipe drawing utils and hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Presentation related imports
import pyautogui
import time

# Presentation moving forward and backward code
def forward_slide():
    pyautogui.press('right')
    time.sleep(0.1)  # Add a delay of 0.1 seconds

def backward_slide():
    pyautogui.press('left')
    time.sleep(0.1)  # Add a delay of 0.1 seconds

# Global variables to store the latest recognized landmarks and gesture
latest_landmarks = []
latest_gesture = None
previous_gesture = None

# Define the callback function for processing results
def print_result(result, image, timestamp):
    global latest_landmarks, latest_gesture, previous_gesture
    if result.gestures:
        top_gesture = result.gestures[0][0]
        latest_gesture = top_gesture.category_name
        if latest_gesture != previous_gesture:
            print(f"Detected gesture: {top_gesture.category_name} with confidence {top_gesture.score:.2f}")
            if top_gesture.category_name == 'Pointing_Up':
                forward_slide()
            elif top_gesture.category_name == 'Thumb_Down':
                backward_slide()
            previous_gesture = latest_gesture
    else:
        latest_gesture = None
        previous_gesture = None

    if result.hand_landmarks:
        latest_landmarks = result.hand_landmarks
    else:
        latest_landmarks = []

# Initialize the GestureRecognizer with live stream mode
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create the GestureRecognizer instance
with vision.GestureRecognizer.create_from_options(options) as recognizer:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Hands
            results = hands.process(frame_rgb)
            
            # Draw the hand landmarks if available
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Convert the frame to MediaPipe image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Perform gesture recognition
            recognizer.recognize_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

            # Display the gesture name on the frame if available
            if latest_gesture:
                cv2.putText(frame, f'Gesture: {latest_gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the image with landmarks
            cv2.imshow('Gesture Recognition', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()
