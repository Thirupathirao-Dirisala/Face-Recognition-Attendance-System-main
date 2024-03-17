import pickle
import os
import face_recognition
def load_images_from_directory(folder_path):
    images = []
    encodings = []
    names = []
    try:
        with open('encodings.pkl', 'rb') as f:
            known_encodings, known_names = pickle.load(f)
    except:
        known_names=[]
    for root, _, files in os.walk(folder_path):
        # Process images from this subfolder
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Case-insensitive image extensions
                image_path = os.path.join(root, filename)
                name = os.path.splitext(filename)[0]
                print(known_names)
                print(name)
                if name not in known_names:
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encoding = face_recognition.face_encodings(image)[0]
                        name = os.path.splitext(filename)[0]  # Extract name from filename (without extension)
                        images.append(image)
                        encodings.append(encoding)
                        names.append(name)
                    except FileNotFoundError:
                        print(f"Error: File not found: {image_path}")  # Handle missing files gracefully
                    except IndexError:
                        print(f"Error: No faces detected in image: {image_path}")  # Handle empty images
    with open('encodings.pkl', 'ab') as f:
        pickle.dump((encodings, names), f)
    return images, encodings, names
load_images_from_directory(f'static/faces')
