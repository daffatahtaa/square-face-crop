# square-face-crop

#### tldr: A python script to batch crop a face on a directory / a folder

# Face Detection and Clustering Tool

This tool is designed to detect faces in images located within a specified directory (including subdirectories), cluster them, and save each face in a separate folder. The tool utilizes the dlib and face_recognition libraries for accurate face detection and clustering using DBSCAN.

## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- Python 3.x
- OpenCV (cv2)
- dlib
- face_recognition
- tqdm
- numpy
- scikit-learn
- Pillow (PIL)

You can install the required dependencies using the following command:

```
pip install opencv-python dlib face_recognition tqdm numpy scikit-learn Pillow
```

## How to Use

1. Clone or download this script and navigate to the script directory.

2. Run the script with the following command:

```
python script.py
```

3. The script will prompt you to enter the path of the folder containing the images you want to process. Provide the path and press Enter.

4. The tool will automatically detect faces in the images, cluster them, and save each face in separate folders within the "faces" directory, which will be created in the same input folder.

5. The output folder structure will be as follows:

```
- input_folder
  - faces
    - person_0
      - image1_face.jpg
      - image2_face.jpg
      ...
    - person_1
      - image3_face.jpg
      - image4_face.jpg
      ...
    ...
```

Note: If no faces are detected in the images, the script will log a warning and no output will be generated.

## Additional Notes

- The minimum size of face images after resizing is set to 768 pixels (adjustable in the script).
- The script logs the process and results in a "script_log.txt" file within the script directory.

Feel free to customize and use this script according to your requirements. If you have any questions or face any issues, please let me know.

Enjoy face detection and clustering with ease!
