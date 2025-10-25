# ---------------------------
# Imports with error handling
# ---------------------------
try:
    import cv2
    import mediapipe as mp
    import numpy as np
    import time
    from math import acos, degrees
except Exception as e:
    print(f"[ERROR] Missing or failed import: {e}")
    print("Install required packages: pip install opencv-python mediapipe numpy")
    raise SystemExit(1)

try:
    import pyttsx3
    tts = pyttsx3.init()
    tts.setProperty('rate', 150)
except Exception:
    tts = None

# ---------------------------
# Helper Functions
# ---------------------------
def angle_between(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosang = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosang = np.clip(cosang, -1.0, 1.0)
    return degrees(acos(cosang))

# ---------------------------
# Pose Angle Definitions
# ---------------------------
L = mp.solutions.pose.PoseLandmark
ANGLE_JOINTS = {
    "left_knee": (L.LEFT_HIP.value, L.LEFT_KNEE.value, L.LEFT_ANKLE.value),
    "right_knee": (L.RIGHT_HIP.value, L.RIGHT_KNEE.value, L.RIGHT_ANKLE.value),
    "left_elbow": (L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_WRIST.value),
    "right_elbow": (L.RIGHT_SHOULDER.value, L.RIGHT_ELBOW.value, L.RIGHT_WRIST.value),
    "left_hip": (L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.LEFT_KNEE.value),
    "right_hip": (L.RIGHT_SHOULDER.value, L.RIGHT_HIP.value, L.RIGHT_KNEE.value),
    "torso": (L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.RIGHT_HIP.value),
}

def compute_angles(landmarks):
    angles = {}
    for name, (ia, ib, ic) in ANGLE_JOINTS.items():
        a = (landmarks[ia].x, landmarks[ia].y)
        b = (landmarks[ib].x, landmarks[ib].y)
        c = (landmarks[ic].x, landmarks[ic].y)
        angles[name] = angle_between(a, b, c)
    return angles

# ---------------------------
# Ideal Pose Angles for Accuracy
# ---------------------------
IDEAL_POSES = {
    # 14 Common Yoga Poses
    "Bound Angle Pose": {"left_knee": 60, "right_knee": 60, "torso": 90},
    "Cat Cow": {"torso": 40},
    "Chair Pose": {"left_knee": 90, "right_knee": 90, "torso": 35},
    "Cobra": {"torso": 170, "left_elbow": 160, "right_elbow": 160},
    "Downward Dog": {"torso": 30, "left_knee": 180, "right_knee": 180},
    "Forward Bend": {"torso": 45, "left_knee": 180, "right_knee": 180},
    "Garland": {"left_knee": 90, "right_knee": 90, "torso": 50},
    "High Lunge": {"left_knee": 160, "right_knee": 90, "torso": 90},
    "Plank": {"torso": 170, "left_elbow": 160, "right_elbow": 160},
    "Reverse Warrior": {"left_knee": 90, "right_knee": 90, "torso": 70},
    "3 Leg Downward Dog": {"torso": 40, "left_knee": 180, "right_knee": 160},
    "Tree Pose": {"left_knee": 180, "right_knee": 60, "torso": 90},
    "Triangle": {"torso": 60, "left_hip": 45, "right_hip": 45},
    "Seated Spinal Twist": {"torso": 70},

    # 12 Surya Namaskar Poses
    "Surya 1: Pranamasana": {"torso": 170, "left_elbow": 0, "right_elbow": 0},
    "Surya 2: Hasta Uttanasana": {"torso": 170, "left_elbow": 170, "right_elbow": 170},
    "Surya 3: Padahastasana": {"torso": 30, "left_knee": 180, "right_knee": 180},
    "Surya 4: Ashwa Sanchalanasana": {"left_knee": 90, "right_knee": 180, "torso": 90},
    "Surya 5: Dandasana": {"torso": 170, "left_elbow": 160, "right_elbow": 160},
    "Surya 6: Ashtanga Namaskara": {"torso": 100, "left_elbow": 85, "right_elbow": 85},
    "Surya 7: Bhujangasana": {"torso": 160, "left_elbow": 150, "right_elbow": 150},
    "Surya 8: Parvatasana": {"torso": 35, "left_knee": 180, "right_knee": 180},
    "Surya 9: Ashwa Sanchalanasana": {"left_knee": 180, "right_knee": 90, "torso": 90},
    "Surya 10: Padahastasana": {"torso": 30, "left_knee": 180, "right_knee": 180},
    "Surya 11: Hasta Uttanasana": {"torso": 170, "left_elbow": 170, "right_elbow": 170},
    "Surya 12: Tadasana": {"torso": 170, "left_knee": 180, "right_knee": 180},
}

# ---------------------------
# Pose Benefits
# ---------------------------
POSE_BENEFITS = {
    # 14 Common Yoga Poses
    "Bound Angle Pose": ["Opens hips and groin", "Improves posture", "Calms the mind"],
    "Cat Cow": ["Increases spine flexibility", "Massages internal organs", "Improves posture"],
    "Chair Pose": ["Strengthens thighs and ankles", "Improves balance", "Tones core muscles"],
    "Cobra": ["Strengthens spine", "Opens chest", "Relieves fatigue"],
    "Downward Dog": ["Stretches hamstrings", "Strengthens arms", "Calms the mind"],
    "Forward Bend": ["Stretches hamstrings", "Relieves stress", "Improves digestion"],
    "Garland": ["Strengthens lower body", "Opens hips", "Stimulates digestion"],
    "High Lunge": ["Strengthens legs", "Opens hips", "Improves balance"],
    "Plank": ["Strengthens core", "Tones arms", "Improves posture"],
    "Reverse Warrior": ["Opens chest and shoulders", "Strengthens legs", "Improves stamina"],
    "3 Leg Downward Dog": ["Stretches hamstrings and calves", "Strengthens arms", "Improves balance"],
    "Tree Pose": ["Improves balance", "Strengthens legs", "Calms the mind"],
    "Triangle": ["Stretches spine and hips", "Strengthens legs", "Improves digestion"],
    "Seated Spinal Twist": ["Improves spine flexibility", "Massages internal organs", "Relieves stress"],

    # 12 Surya Namaskar
    "Surya 1: Pranamasana": ["Improves posture", "Calms the mind", "Warms up body"],
    "Surya 2: Hasta Uttanasana": ["Stretches spine", "Opens chest", "Improves digestion"],
    "Surya 3: Padahastasana": ["Stretches hamstrings", "Strengthens legs", "Relieves stress"],
    "Surya 4: Ashwa Sanchalanasana": ["Opens hips", "Strengthens legs", "Improves balance"],
    "Surya 5: Dandasana": ["Strengthens core", "Improves posture", "Tones arms"],
    "Surya 6: Ashtanga Namaskara": ["Strengthens arms and legs", "Improves spine flexibility", "Activates core muscles"],
    "Surya 7: Bhujangasana": ["Strengthens spine", "Opens chest", "Relieves fatigue"],
    "Surya 8: Parvatasana": ["Stretches spine and hamstrings", "Strengthens arms", "Improves balance"],
    "Surya 9: Ashwa Sanchalanasana": ["Opens hips", "Strengthens legs", "Improves balance"],
    "Surya 10: Padahastasana": ["Stretches hamstrings", "Relieves stress", "Improves digestion"],
    "Surya 11: Hasta Uttanasana": ["Stretches spine", "Opens chest", "Improves digestion"],
    "Surya 12: Tadasana": ["Improves posture", "Strengthens legs", "Calms the mind"],
}

# ---------------------------
# Accuracy Evaluation
# ---------------------------
def evaluate_accuracy(pose_name, angles):
    if pose_name not in IDEAL_POSES:
        return 0
    ideal = IDEAL_POSES[pose_name]
    total, count = 0, 0
    for joint, ideal_angle in ideal.items():
        if joint in angles:
            diff = abs(angles[joint] - ideal_angle)
            score = max(0, 100 - diff)
            total += score
            count += 1
    return round(total / count, 1) if count else 0

# ---------------------------
# Pose Detection Functions
# ---------------------------
def detect_yoga_pose(angles):
    pose_name = "Unknown"
    issues = []

    if angles["left_knee"] < 120 and angles["right_knee"] < 120 and angles["left_hip"] > 50 and angles["right_hip"] > 50:
        pose_name = "Bound Angle Pose"
        issues.append("Keep your back straight and knees relaxed outward.")
    elif angles["torso"] < 50:
        pose_name = "Cat Cow"
        issues.append("Move your spine in sync with breath.")
    elif angles["left_knee"] < 120 and angles["right_knee"] < 120 and angles["torso"] < 40:
        pose_name = "Chair Pose"
        issues.append("Keep your back more upright.")
    elif angles["torso"] > 160 and angles["left_elbow"] > 150 and angles["right_elbow"] > 150:
        pose_name = "Cobra"
    elif angles["torso"] < 45 and angles["left_knee"] > 150 and angles["right_knee"] > 150:
        pose_name = "Downward Dog"
        issues.append("Straighten your spine and legs more.")
    elif angles["torso"] < 60 and angles["left_knee"] > 150 and angles["right_knee"] > 150:
        pose_name = "Forward Bend"
    elif angles["left_knee"] < 120 and angles["right_knee"] < 120 and angles["torso"] < 50:
        pose_name = "Garland"
    elif angles["left_knee"] > 140 and angles["right_knee"] < 120:
        pose_name = "High Lunge"
    elif angles["left_elbow"] > 150 and angles["right_elbow"] > 150 and angles["torso"] > 160:
        pose_name = "Plank"
    elif angles["left_knee"] < 120 and angles["right_knee"] < 120 and angles["torso"] > 50:
        pose_name = "Reverse Warrior"
    elif angles["torso"] < 45 and (angles["left_knee"] < 170 or angles["right_knee"] < 170):
        pose_name = "3 Leg Downward Dog"
    elif (angles["left_knee"] < 140 and angles["right_knee"] > 170) or (angles["right_knee"] < 140 and angles["left_knee"] > 170):
        pose_name = "Tree Pose"
        issues.append("Keep your supporting leg straight.")
    elif angles["torso"] < 90 and (angles["left_hip"] < 60 or angles["right_hip"] < 60):
        pose_name = "Triangle"
    elif angles["torso"] < 80:
        pose_name = "Seated Spinal Twist"

    return pose_name, issues

def detect_surya_pose(angles):
    pose_name = "Unknown"
    issues = []

    # Surya 1
    if angles["torso"] > 160 and angles["left_elbow"] < 20 and angles["right_elbow"] < 20:
        pose_name = "Surya 1: Pranamasana"
        issues.append("Stand straight, feet together, palms joined at chest.")
    # Surya 2
    elif angles["torso"] > 160 and angles["left_elbow"] > 150 and angles["right_elbow"] > 150:
        pose_name = "Surya 2: Hasta Uttanasana"
        issues.append("Stretch arms overhead, arch slightly backward, gaze upward.")
    # Surya 3
    elif angles["torso"] < 50 and angles["left_knee"] > 160 and angles["right_knee"] > 160:
        pose_name = "Surya 3: Padahastasana"
        issues.append("Keep legs straight, bend forward from hips, try touching toes.")
    # Surya 4
    elif 80 <= angles["left_knee"] <= 100 and angles["right_knee"] > 160 and angles["torso"] < 120:
        pose_name = "Surya 4: Ashwa Sanchalanasana"
        issues.append("Front knee at 90°, back leg straight, gaze forward, hips squared.")
    # Surya 5
    elif angles["torso"] > 160 and angles["left_elbow"] > 150 and angles["right_elbow"] > 150:
        pose_name = "Surya 5: Dandasana"
        issues.append("Body straight, core tight, hands under shoulders.")
    # Surya 6
    torso_ok = 90 <= angles["torso"] <= 110
    elbows_ok = 70 <= angles["left_elbow"] <= 100 and 70 <= angles["right_elbow"] <= 100
    if torso_ok and elbows_ok:
        pose_name = "Surya 6: Ashtanga Namaskara"
        issues.append("Chest and chin on floor, hips slightly raised, elbows close to body.")
    # Surya 7
    elif 150 <= angles["torso"] <= 180 and 140 <= angles["left_elbow"] <= 170 and 140 <= angles["right_elbow"] <= 170:
        pose_name = "Surya 7: Bhujangasana"
        issues.append("Lift chest, shoulders relaxed, elbows slightly bent.")
    # Surya 8
    elif 20 <= angles["torso"] <= 45 and 160 <= angles["left_knee"] <= 180 and 160 <= angles["right_knee"] <= 180:
        pose_name = "Surya 8: Parvatasana"
        issues.append("Form inverted V, hands and feet pressing, spine straight.")
    # Surya 9
    elif 80 <= angles["right_knee"] <= 100 and angles["left_knee"] > 160 and angles["torso"] < 120:
        pose_name = "Surya 9: Ashwa Sanchalanasana"
        issues.append("Front knee at 90°, back leg straight, gaze forward.")
    # Surya 10
    elif angles["torso"] < 50 and angles["left_knee"] > 160 and angles["right_knee"] > 160:
        pose_name = "Surya 10: Padahastasana"
        issues.append("Bend forward from hips, legs straight, try touching toes.")
    # Surya 11
    elif angles["torso"] > 160 and angles["left_elbow"] > 150 and angles["right_elbow"] > 150:
        pose_name = "Surya 11: Hasta Uttanasana"
        issues.append("Stretch arms overhead, arch slightly backward.")
    # Surya 12
    elif angles["torso"] > 160 and angles["left_knee"] > 170 and angles["right_knee"] > 170:
        pose_name = "Surya 12: Tadasana"
        issues.append("Stand tall, feet together, shoulders relaxed.")

    return pose_name, issues

# ---------------------------
# Webcam Loop
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    raise SystemExit(1)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

print("Starting webcam. Press 'q' to quit.")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_estimator:
    last_speak_time = 0
    SPEAK_COOLDOWN = 3

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(img_rgb)

        display_label = "No Pose"
        issues = []
        accuracy = 0
        benefits = []

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            angles = compute_angles(results.pose_landmarks.landmark)

            # Detect yoga poses first
            display_label, issues = detect_yoga_pose(angles)
            if display_label == "Unknown":
                display_label, issues = detect_surya_pose(angles)

            accuracy = evaluate_accuracy(display_label, angles)
            benefits = POSE_BENEFITS.get(display_label, [])

            # Voice feedback
            if issues and tts:
                now = time.time()
                if now - last_speak_time > SPEAK_COOLDOWN:
                    try:
                        tts.say(issues[0])
                        tts.runAndWait()
                    except Exception:
                        pass
                    last_speak_time = now

        # Overlay info
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 160), (0, 0, 0), -1)
        cv2.putText(frame, f"Pose: {display_label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if issues:
            cv2.putText(frame, f"Issue: {issues[0]}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Accuracy: {accuracy}%", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if benefits:
            cv2.putText(frame, f"Benefits: {', '.join(benefits[:3])}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Accuracy meter bar
       

        cv2.imshow("Yoga Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
if tts:
    try:
        tts.stop()
    except Exception:
        pass
