import cv2
import face_recognition
import os
import numpy as np
import dlib
import face_recognition_models

# ✅ Load model paths manually (for new versions)
predictor_path = face_recognition_models.pose_predictor_model_location()
model_path = face_recognition_models.face_recognition_model_location()

# --- Setup dlib manually (to avoid API mismatch) ---
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(model_path)

# --- Load known faces ---
path = "dataset"
known_faces = []
known_names = []

print("⏳ Loading face encodings...")

for name in os.listdir(path):
    person_folder = os.path.join(path, name)
    if not os.path.isdir(person_folder):
        continue

    for filename in os.listdir(person_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(person_folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dets = detector(rgb_image, 1)

            for d in dets:
                shape = sp(rgb_image, d)
                face_descriptor = facerec.compute_face_descriptor(rgb_image, shape)
                known_faces.append(np.array(face_descriptor))
                known_names.append(name)

print(f"✅ Loaded {len(known_faces)} face encodings.")

# --- Initialize webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    dets = detector(rgb_small, 1)
    face_encodings = []

    for d in dets:
        shape = sp(rgb_small, d)
        face_descriptor = facerec.compute_face_descriptor(rgb_small, shape)
        face_encodings.append(np.array(face_descriptor))

    # Compare faces
    for d, face_encoding in zip(dets, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.45)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_names[best_match_index]

        # Scale back to original frame size
        top = int(d.top() * 4)
        right = int(d.right() * 4)
        bottom = int(d.bottom() * 4)
        left = int(d.left() * 4)

        # Draw rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()