import streamlit as st
import cv2
import dlib
import numpy as np
import torch
from PIL import Image
from helper import identify_face, create_embeddings, known_faces_directory, inference, eye_aspect_ratio, draw_face_frame, crop_to_face_frame, evaluate_model_on_test_data
from datasets import AnhDangDataset, TRAIN_PATH, VAL_PATH, TEST_PATH
from torch.utils.data import DataLoader
from models import FaceVerificationModel
import time
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Initialize model and load known face embeddings
model = FaceVerificationModel(embedding_size=128)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.load_state_dict(torch.load('models/pretrained/best_model.pt', map_location=device))
model.eval()
known_embeddings = create_embeddings(model, known_faces_directory)

predictor_path = "app/predictor/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

st.title("Liveness Detection and Face Recognition Demo")
FRAME_WINDOW = st.image([])
status_text = st.empty()
EYE_AR_THRESH = 0.2  # Threshold for EAR
blink_counter = 0
verified = False
image_captured = False

def verify():
    global verified, blink_counter, image_captured
    cap = cv2.VideoCapture(0)

    while st.session_state.verification_in_progress and not verified:
        ret, frame = cap.read()
        if not ret:
            status_text.text('Failed to capture image.')
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = detector(gray, 0)
        frame_with_box, top_left, bottom_right = draw_face_frame(frame_rgb)
        FRAME_WINDOW.image(frame_with_box)

        handle_faces(faces, frame_rgb, gray, top_left, bottom_right, cap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    FRAME_WINDOW.empty()

def handle_faces(faces, frame_rgb, gray, top_left, bottom_right, cap):
    global verified, blink_counter
    if len(faces) == 0:
        status_text.text("No human detected. Please ensure you are in the camera frame.")
        verified = False
    elif len(faces) > 1:
        status_text.text("Multiple faces detected. Please ensure only one person is in the camera frame.")
        verified = False
    else:
        face = faces[0]
        face_top_left = (face.left(), face.top())
        face_bottom_right = (face.right(), face.bottom())
        shape = predictor(gray, face)
        points = np.array([[p.x, p.y] for p in shape.parts()])
        leftEye = points[42:48]
        rightEye = points[36:42]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        if ear < EYE_AR_THRESH:
            blink_counter += 1
        if blink_counter >= 3:
            verified = True
            blink_counter = 0
            cap.release()  # Release camera immediately after verification
            cv2.destroyAllWindows()
            FRAME_WINDOW.empty()
            capture_and_process_image(frame_rgb, top_left, bottom_right, face_top_left, face_bottom_right)

def capture_and_process_image(frame_rgb, top_left, bottom_right, face_top_left, face_bottom_right):
    status_text.text("Human verified successfully! Capturing image now...")
    time.sleep(1)  # Short sleep to simulate capturing delay
    perform_final_operations(frame_rgb, top_left, bottom_right, face_top_left, face_bottom_right)

def perform_final_operations(frame_rgb, top_left, bottom_right, face_top_left, face_bottom_right):
    global image_captured
    cropped_frame = crop_to_face_frame(frame_rgb, top_left, bottom_right)
    cropped_face = crop_to_face_frame(frame_rgb, face_top_left, face_bottom_right)
    cv2.imwrite('app/saved/captured_face.jpg', cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite('app/saved/cropped_face.jpg', cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
    evaluate_captured_image(cropped_frame)

def evaluate_captured_image(cropped_frame):
    global verified, image_captured
    pil_image, fer_dict_str, deepfake_dict_str, va_dict_str = inference('app/saved/captured_face.jpg')
    print(f'Deepfake detection: {deepfake_dict_str}')
    if deepfake_dict_str is None or not deepfake_dict_str:
        print("Deepfake detection result is not available.")
        status_text.text("Deepfake detection result is not available.")
        verified = False
        image_captured = False
    elif any(label == 'Fake' for label in eval(deepfake_dict_str).values()):
        print(f'Deepfake detected: {deepfake_dict_str}')
        status_text.text("Spoofing detected. Please try again.")
        verified = False
        image_captured = False
    else:
        identify_and_display_results(pil_image, fer_dict_str, deepfake_dict_str, va_dict_str, cropped_frame)

def identify_and_display_results(pil_image, fer_dict_str, deepfake_dict_str, va_dict_str, cropped_frame):
    identified_person = identify_face(model, 'app/saved/captured_face.jpg', known_embeddings, threshold=0.5, distance_metric='chebyshev') # cosine, euclidean, manhattan, chebyshev
    status_text.text(f"Identification complete: {identified_person}")

    col1, col2 = st.columns(2)
    with col1:
        st.image(cropped_frame, caption="Captured Image", use_column_width=True, width=500)
    with col2:
        st.image(pil_image, caption="Face Detection and 3D Landmarks", use_column_width=True, width=500)

    st.write(f"**Facial Expression Recognition:** {eval(fer_dict_str)}")
    st.write(f"**Facial Valence Arousal:** {eval(va_dict_str)}")
    st.write(f"**DeepFake Detection:** {eval(deepfake_dict_str)}")
    image_captured = True

def save_face_image(name, image):
    # Ensure the directory exists
    if not os.path.exists(known_faces_directory):
        os.makedirs(known_faces_directory)
    file_path = os.path.join(known_faces_directory, f"{name}.jpg")
    cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if 'verification_in_progress' not in st.session_state:
    st.session_state.verification_in_progress = False

# Create buttons and center them
if st.button('Verify Identity'):
    st.session_state.verification_in_progress = True
    verify()

if st.button('Reset'):
    st.session_state.verification_in_progress = False
    FRAME_WINDOW.empty()
    status_text.text("")
    st.experimental_rerun()

# Registration page
st.sidebar.title("Register New Face")
new_face_name = st.sidebar.text_input("Enter new face name:", key="new_face_name_input")

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="file_uploader")

if st.sidebar.button("Register Face"):
    if uploaded_file is not None and new_face_name:
        image = Image.open(uploaded_file)
        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = detector(gray, 0)
        if len(faces) == 1:
            face = faces[0]
            top_left = (face.left(), face.top())
            bottom_right = (face.right(), face.bottom())
            cropped_frame = crop_to_face_frame(image, top_left, bottom_right)
            save_face_image(new_face_name, cropped_frame)
            st.sidebar.success(f"New face '{new_face_name}' registered successfully!")
        else:
            st.sidebar.error("Failed to detect a single face. Please try again with a different image.")
    else:
        st.sidebar.error("Please enter a name and upload an image.")

# ROC/AUC evaluation page
st.sidebar.title("Model Testing")

# Load the test dataset and create DataLoader
default_max_samples = 10
max_samples = st.sidebar.number_input('Max Samples', min_value=1, value=default_max_samples)
test_dataset = AnhDangDataset(TEST_PATH, max_samples=max_samples)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32)

# Evaluate model on the test dataset
predictions, valid_labels = evaluate_model_on_test_data(model, test_loader)

fpr, tpr, _ = roc_curve(valid_labels, predictions)
roc_auc = auc(fpr, tpr)

st.sidebar.subheader("ROC Curve")
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
st.sidebar.pyplot(plt)