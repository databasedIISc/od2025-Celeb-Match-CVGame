import cv2
import numpy as np
import pickle
from PIL import Image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.all_utils import read_yaml
import os
import random

# Load configurations
config = read_yaml('config/config.yaml')
params = read_yaml('params.yaml')

artifacts = config['artifacts']
artifacts_dir = artifacts['artifacts_dir']

# Feature extraction paths
feature_extraction_dir = artifacts['feature_extraction_dir']
extracted_features_name = artifacts['extracted_features_name']
feature_extraction_path = os.path.join(artifacts_dir, feature_extraction_dir)
features_name = os.path.join(feature_extraction_path, extracted_features_name)

# Load precomputed features and filenames
feature_list = pickle.load(open(features_name, 'rb'))
filenames = pickle.load(open(os.path.join(artifacts_dir, artifacts['pickle_format_data_dir'], artifacts['img_pickle_file_name']), 'rb'))

# Load model
detector = MTCNN()
model = VGGFace(model=params['base']['BASE_MODEL'], include_top=params['base']['include_top'], 
                input_shape=(224,224,3), pooling=params['base']['pooling'])

# Function to extract features from a frame
def extract_features_from_frame(frame, model, detector):
    results = detector.detect_faces(frame)
    if results:
        x, y, width, height = results[0]['box']
        face = frame[y:y + height, x:x + width]
        image = cv2.resize(face, (224, 224))
        face_array = np.asarray(image).astype('float32')
        expanded_img = np.expand_dims(face_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        return model.predict(preprocessed_img).flatten(), (x, y, width, height)
    return None, None

# Function to find the best match
def find_best_match(features, feature_list):
    similarity = [cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0] for i in range(len(feature_list))]
    best_match_index = np.argmax(similarity)
    return filenames[best_match_index], similarity[best_match_index]

# Start webcam
cap = cv2.VideoCapture(0)
frames_without_match = 0
frames_with_match = 0
max_no_match_frames = 800
match_display_frames = 15
predicted_name = None
capture_frame = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam.")
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if capture_frame:
        features, bbox = extract_features_from_frame(frame_rgb, model, detector)
        img = None
        if features is not None:
            best_match, confidence = find_best_match(features, feature_list)
            predicted_name = " ".join(best_match.split('\\')[1].split('_'))
            frames_with_match = match_display_frames  # Keep name for 20 frames
            frames_without_match = 0  # Reset counter

            img_match = cv2.imread(best_match)
            idx = best_match.split('\\')
            nm = idx[1]
            num = idx[2].split('.')[1]

            full_img_path = f"archive/{nm}/{num}.jpg"
            img = cv2.imread(full_img_path)
            if img is None:
                # Iterate through the directory to find the first non-None image
                files = os.listdir(f"archive/{nm}")
                random.shuffle(files)
                for file in files:
                    img = cv2.imread(os.path.join(f"archive/{nm}", file))
                    if img is not None:
                        break

            # Scale the image without distorting it
            img = cv2.resize(img, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
            frames_without_match += 1

        if frames_with_match > 0 and bbox:
            x, y, width, height = bbox
            # cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            match_percentage = confidence * 100
            cv2.putText(frame, f"{predicted_name} ({match_percentage:.2f}%)", (x, y + height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frames_with_match -= 1

        if img is not None:
            combined_frame = np.hstack((frame, img))
            cv2.imshow("Webcam Feed", combined_frame)
        else:
            cv2.imshow("Webcam Feed", frame)
            

        if frames_without_match >= max_no_match_frames:
            print("Error: Fault in system - No match detected in 1000 frames")
            break

        # Pause the camera feed
        while capture_frame:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                capture_frame = False
                break
            elif key == ord('q'):
                capture_frame = False
                break
            if img is not None:
                cv2.imshow("Webcam Feed", combined_frame)
            else:
                cv2.imshow("Webcam Feed", frame)    

    cv2.imshow("Webcam Feed", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        capture_frame = True
    elif key == ord('r'):
        capture_frame = False

cap.release()
cv2.destroyAllWindows()
