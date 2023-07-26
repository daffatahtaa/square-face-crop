import os
import cv2
import dlib
import face_recognition
from tqdm import tqdm
import logging
import numpy as np
import shutil
from sklearn.cluster import DBSCAN
from PIL import Image
from pathlib import Path as path

def setup_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a file handler and set the logging level with 'utf-8' encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and attach it to the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

def are_encodings_equal(encoding1, encoding2):
    if len(encoding1) != len(encoding2):
        return False

    distance = np.linalg.norm(np.array(encoding1) - np.array(encoding2))
    return distance < 0.8  # Adjust the threshold value for similarity
    
def detect_and_save_faces(input_path: str, output_dir: str = "faces", min_image_size: int = 768):
    """
    Detect faces in the images located in the 'input_path' directory, including subdirectories,
    cluster them, and save each face in a separate folder in the 'output_dir'.

    Parameters:
        input_path (str): The directory containing the images to process.
        output_dir (str): The directory where the faces will be saved. Default is "faces".
        min_image_size (int): The minimum size (in pixels) of the face images after resizing.

    Returns:
        None
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")
    
    if os.path.exists(input_path):
        print (f"\n Pass {input_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    invalid_dir = os.path.join(output_dir, "invalid_images")
    os.makedirs(invalid_dir, exist_ok=True)

    image_files = []
    for root, _, files in os.walk(input_path):
        if root != os.path.join(input_path, output_dir):
            image_files.extend([os.path.join(root, f) for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    face_encodings = []
    face_locations = []
    face_image_paths = []
    
    for image_file in tqdm(image_files, desc=f"Scanning faces in images"):
        # image = cv2.imread(image_file)
        
        # Read non ASCII 
        stream = open(image_file, 'rb')
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype = np.uint8)
        image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        current_face_locations = face_recognition.face_locations(rgb_image, model="hog")
        current_face_encodings = face_recognition.face_encodings(rgb_image, current_face_locations)

        face_encodings.extend(current_face_encodings)
        face_locations.extend(current_face_locations)
        face_image_paths.extend([image_file] * len(current_face_locations))

    if not face_encodings:
        logger.warning(f"No faces detected in {input_path} images.")
        return

    clustering_model = DBSCAN(eps=0.5, min_samples=2, metric="euclidean")
    labels = clustering_model.fit_predict(face_encodings)

    logger.info("Processing images and saving faces...")
    for label, face_location, image_path in tqdm(zip(labels, face_locations, face_image_paths), total=len(labels)):
        top, right, bottom, left = face_location
            
        padding = 50
        top = max(top - padding, 0)
        left = max(left - padding, 0)
        bottom = min(bottom + padding, rgb_image.shape[0])
        right = min(right + padding, rgb_image.shape[1])
        
        # Additional check to ensure the face region is valid
        if top >= bottom or left >= right:
            logger.warning(f"Invalid face location detected in {image_path}. Skipping...")
            try:
                shutil.copy(image_path, os.path.join(invalid_dir, os.path.basename(image_path)))
            except:
                logging.warning("FILE EXIST")
                continue
            continue
        
        # image = cv2.imread(image_path)
        
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_image = rgb_image[top:bottom, left:right]

        face_image_pil = Image.fromarray(face_image)
        width, height = face_image_pil.size
        
        if width < min_image_size or height < min_image_size:
            
            # Maintain the aspect ratio and resize the image to be square
            # new_size = (min(width, height), min(width, height))
            # face_image_pil = face_image_pil.resize(new_size, Image.LANCZOS)
            # face_image = np.array(face_image_pil)
            
            new_width = max(min_image_size, width)
            new_height = max(min_image_size, height)
            aspect_ratio = float(width) / float(height)
            
            if width < height:
                new_width = int(new_height * aspect_ratio)
            else:
                new_height = int(new_width * aspect_ratio)
            
            face_image_pil = face_image_pil.resize((new_width, new_height), Image.LANCZOS)
            face_image = np.array(face_image_pil)

        person_dir = os.path.join(input_path, output_dir, f"person_{label}")

        os.makedirs(person_dir, exist_ok=True)

        output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_face.jpg"
        output_filepath = os.path.join(person_dir, output_filename)

        output_filepath_encoded = output_filepath.encode('utf-8')
        # cv2.imwrite(output_filepath, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        cv2.imencode('.jpg', cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))[1].tofile(output_filepath_encoded)
        
        logger.info(f"A face image has been detected {image_path}, as a{person_dir}")

    logging.info("Face detection and saving completed.")

if __name__ == "__main__":
    log_file = "script_log.txt"
    # input_folder = input("Enter the path of the folder containing images: ")

    with open(log_file, 'a'):
        pass

    # Set up logging
    logger = setup_logger(log_file)

    while True:
        input_folder = input("Enter the path of the folder containing images (or type 'exit' to quit): ")
        # input_folder = input_folder.encode('utf-16').decode('utf-16')
        output_folder = os.path.join(input_folder, "faces")
        
        if input_folder.lower() == "exit":
            break

        output_folder = os.path.join(input_folder, "faces")

        # Log the start of the script
        logger.info("Script started.")

        detect_and_save_faces(input_folder, output_folder)

        # Log the end of the script
        logger.info("Script finished.")

    print("Goodbye!")
