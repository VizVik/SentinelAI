import cv2
import mediapipe as mp
import math
from config import *

closed_frames = 0
yawn_frames = 0
fatigue_score = 0

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
    if eye_width == 0:
        return 0

    ear = eye_height / eye_width

    return ear
def calculate_mar(face_landmarks):

    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]

    left_corner = face_landmarks.landmark[78]
    right_corner = face_landmarks.landmark[308]

    mouth_height = euclidean_distance(
        (upper_lip.x, upper_lip.y),
        (lower_lip.x, lower_lip.y)
    )

    mouth_width = euclidean_distance(
        (left_corner.x, left_corner.y),
        (right_corner.x, right_corner.y)
    )
    if mouth_width == 0:
        return 0

    mar = mouth_height / mouth_width

    return mar

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
        mar = calculate_mar(face_landmarks)
        fatigue_score = 0


        if ear < EAR_THRESHOLD:
            closed_frames += 1
        else:
            closed_frames = 0
        if closed_frames >= DROWSY_FRAMES:
            fatigue_score += 70
        if mar > MAR_THRESHOLD:
            yawn_frames += 1
        else:
            yawn_frames = 0
        if yawn_frames >= YAWN_FRAMES:
            fatigue_score += 30

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Frames: {closed_frames}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"MAR: {mar:.3f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2
        )
        cv2.putText(
            frame,
            f"Yawn Frames: {yawn_frames}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2
        )
        cv2.putText(
            frame,
            f"Fatigue Score: {fatigue_score}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2
        )
        if yawn_frames >= YAWN_FRAMES:

            cv2.putText(
                frame,
                "YAWNING",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )
        
        if fatigue_score >= 80:

            status = "DROWSY"
            color = (0, 0, 255)

        elif fatigue_score >= 30:

            status = "FATIGUED"
            color = (0, 255, 255)

        else:

            status = "AWAKE"
            color = (0, 255, 0)

        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        if fatigue_score >= 80:

            cv2.putText(
                frame,
                "DROWSINESS ALERT",
                (50, 210),
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
        "SentinelAI Drowsiness Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()