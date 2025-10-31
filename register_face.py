import cv2
import face_recognition
import pickle
import os

def register_face(username):
    cam = cv2.VideoCapture(0)
    print("\n📸 Capturing face... Press 'q' to capture your face image.\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Failed to access camera.")
            break

        cv2.imshow("Face Registration - Press 'q' to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    # Save captured image
    os.makedirs("faces", exist_ok=True)
    img_path = f"faces/{username}.jpg"
    cv2.imwrite(img_path, frame)
    print(f"✅ Face image saved as {img_path}")

    # Encode face
    image = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print("⚠️ No face detected. Please try again.")
        return

    encoding = encodings[0]

    # Save encoding to file
    data = {username: encoding}
    with open("face_encodings.pkl", "wb") as f:
        pickle.dump(data, f)

    print(f"✅ Face encoding saved successfully for {username}!")

if __name__ == "__main__":
    name = input("Enter your name: ").strip()
    register_face(name)