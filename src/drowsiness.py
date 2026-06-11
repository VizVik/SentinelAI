import cv2
import mediapipe as mp
import math
from config import *

closed_frames = 0

def euclidean_distance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2

    distance = math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )

    return distance
def calculate_ear(face_landmarks):

    left_corner = face_landmarks.landmark[33]
    right_corner = face_landmarks.landmark[133]

    upper_lid = face_landmarks.landmark[159]
    lower_lid = face_landmarks.landmark[145]

    eye_width = euclidean_distance(
        (left_corner.x, left_corner.y),
        (right_corner.x, right_corner.y)
    )

    eye_height = euclidean_distance(
        (upper_lid.x, upper_lid.y),
        (lower_lid.x, lower_lid.y)
    )

    ear = eye_height / eye_width

    return ear

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(
    thickness=1,
    circle_radius=1
)

# Webcam
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        print("Camera not found")
        break

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        ear = calculate_ear(face_landmarks)

        if ear < EAR_THRESHOLD:
            closed_frames += 1
        else:
            closed_frames = 0
        
        if closed_frames >= DROWSY_FRAMES:
            cv2.putText(
                frame,
                "DROWSINESS ALERT",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        for face_landmarks in results.multi_face_landmarks:

            for idx, landmark in enumerate(face_landmarks.landmark):

                h, w, _ = frame.shape

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                if idx in [33, 263, 13, 14]:
                    cv2.putText(
                        frame,
                        str(idx),
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1
                    )

    cv2.imshow(
        "SentinelAI Face Mesh",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()