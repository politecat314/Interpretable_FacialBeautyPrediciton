from typing import Tuple, Union
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from tqdm import tqdm

def get_landmark_coords(image, detection_result, index):
    x = detection_result.face_landmarks[0][index].x
    y = detection_result.face_landmarks[0][index].y
    
    return int(x*image.shape[1]), int(y*image.shape[0])

def convert_coordinates(upper, bottom, left, right):
    x1 = left[0]
    x2 = right[0]
    y1 = upper[1]
    y2 = bottom[1]

    if y1 > y2:
        y1, y2, = y2, y1
    elif y1 == y2:
        y1 -= 15

    return x1, x2, y1, y2

def nose(image, detection_result):
    # keypoints given by the detection_result
    upper = get_landmark_coords(image, detection_result, 8)
    bottom = get_landmark_coords(image, detection_result, 391)
    left = get_landmark_coords(image, detection_result, 165)
    right = get_landmark_coords(image, detection_result, 391)

    return convert_coordinates(upper, bottom, left, right)

def lips(image, detection_result):
    # keypoints given by the detection_result
    upper = get_landmark_coords(image, detection_result, 267)
    bottom = get_landmark_coords(image, detection_result, 17)
    left = get_landmark_coords(image, detection_result, 61)
    right = get_landmark_coords(image, detection_result, 291)
    
    return convert_coordinates(upper, bottom, left, right)

def eyes(image, detection_result):
    # keypoints given by the detection_result
    upper = get_landmark_coords(image, detection_result, 443)
    bottom = get_landmark_coords(image, detection_result, 450)
    left = get_landmark_coords(image, detection_result, 226)
    right = get_landmark_coords(image, detection_result, 446)
    
    return convert_coordinates(upper, bottom, left, right)

def left_eye(image, detection_result):
    # keypoints given by the detection_result
    x1, x2, y1, y2 = eyes(image, detection_result)
    
    return x1, (x1 + x2)//2, y1, y2

def right_eye(image, detection_result):
    # keypoints given by the detection_result
    x1, x2, y1, y2 = eyes(image, detection_result)
    
    return (x1 + x2)//2, x2, y1, y2

def left_cheek(image, detection_result):
    # keypoints given by the detection_result
    upper = get_landmark_coords(image, detection_result, 227)
    bottom = get_landmark_coords(image, detection_result, 140)
    left = get_landmark_coords(image, detection_result, 227)
    right = get_landmark_coords(image, detection_result, 140)

    return convert_coordinates(upper, bottom, left, right)

def right_cheek(image, detection_result):
    # keypoints given by the detection_result
    upper = get_landmark_coords(image, detection_result, 447)
    bottom = get_landmark_coords(image, detection_result, 369)
    left = get_landmark_coords(image, detection_result, 369)
    right = get_landmark_coords(image, detection_result, 447)

    return convert_coordinates(upper, bottom, left, right)

def chin(image, detection_result):
    upper = get_landmark_coords(image, detection_result, 17)
    bottom = get_landmark_coords(image, detection_result, 369)
    left = get_landmark_coords(image, detection_result, 140)
    right = get_landmark_coords(image, detection_result, 369)

    x1, x2, y1, y2 = convert_coordinates(upper, bottom, left, right)
    y2 += 15
    
    return x1, x2, y1, y2

def eyebrows(image, detection_result):
    upper = get_landmark_coords(image, detection_result, 104)
    bottom = get_landmark_coords(image, detection_result, 383)
    left = get_landmark_coords(image, detection_result, 21)
    right = get_landmark_coords(image, detection_result, 251)
    
    return convert_coordinates(upper, bottom, left, right)

def left_eyebrow(image, detection_result):
    x1, x2, y1, y2 = eyebrows(image, detection_result)

    return x1, (x1 + x2)//2, y1, y2

def right_eyebrow(image, detection_result):
    x1, x2, y1, y2 = eyebrows(image, detection_result)

    return (x1 + x2)//2, x2, y1, y2

os.chdir("C:/Users/ugail/Documents/paperV2")

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
landmark_detector = vision.FaceLandmarker.create_from_options(options)

base_options = python.BaseOptions(model_asset_path="blaze_face_short_range.tflite")
options = vision.FaceDetectorOptions(base_options=base_options)
face_detector = vision.FaceDetector.create_from_options(options)

aligner = vision.FaceAligner.create_from_model_path("face_landmarker.task")

names = [
    "nose",
    "lips",
    "eyes",
    "left_eye",
    "right_eye",
    "right_cheek",
    "left_cheek",
    "chin",
    "eyebrows",
    "left_eyebrow",
    "right_eyebrow",
]

names2 = [
    "nose",
    "lips",
    "left_eye",
    "right_eye",
    "right_cheek",
    "left_cheek",
    "chin",
    "left_eyebrow",
    "right_eyebrow",
]

names3 = [
    "nose",
    "lips",
    "eyes",
    "cheeks",
    "chin",
    "eyebrows",
]

functions = [
    nose,
    lips,
    eyes,
    left_eye,
    right_eye,
    right_cheek,
    left_cheek,
    chin,
    eyebrows,
    left_eyebrow,
    right_eyebrow,
]

def align(path):
    aligned_image = aligner.align(
        mp.Image.create_from_file(path)
    )
    return np.array(aligned_image.numpy_view()[...,::-1])

def detect_landmarks(path=None, image=None):
    if path:
        image = align(path)
    detection_result = landmark_detector.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    )
    return image, detection_result

def crop_feature(name, image, detection_result):
    cropper = functions[names.index(name)]
    x1, x2, y1, y2 = cropper(image, detection_result)
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image, np.array([x1, y1]), np.array([x2, y2])

def crop_face(image):
    detection_result = face_detector.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    )
    detection = detection_result.detections[0]
    bbox = detection.bounding_box

    return (
        np.array([
            bbox.origin_x, 
            bbox.origin_y
        ]), 
        np.array([
            bbox.origin_x + bbox.width,
            bbox.origin_y + bbox.height
        ])
    )

def replace_feature(name, image1, image2, detection_result1, detection_result2):
    feature1, p1, p2 = crop_feature(name, image1, detection_result1)
    feature2, _, _ = crop_feature(name, image2, detection_result2)

    feature2 = cv2.resize(feature2, feature1.shape[:-1][::-1], interpolation=cv2.INTER_LANCZOS4)  
    image = image1.copy()
    image[p1[1]:p2[1],p1[0]:p2[0]] = feature2

    return image

def generate_dataset(path, *args):
    for file in tqdm(os.listdir(path)):
        try:
            image, detection_result = detect_landmarks(
                path=os.path.join(path,file)
            )
            for name in args:
                save_path = os.path.join(path.split("/")[0],name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                if name == "mediapipe":
                    m1, m2 = crop_face(image)
                    cropped_image = image[m1[1]:m2[1],m1[0]:m2[0]]
                    cv2.imwrite(os.path.join(save_path,file), cropped_image)

                else:
                    feature, _, _ = crop_feature(name, image, detection_result)
                    cv2.imwrite(os.path.join(save_path,file), feature)
        except:
            print(file)

def check_paths(file, dataset, names):
    if file == "zero.npy":
        return False

    return np.array([
        file in os.listdir(os.path.join(dataset,name))
        for name in names
    ]).all()

def permutation(path, k1, k2, *args):
    dataset = path.split("/")[0]
    for i in range(k1, k2):
        files = np.array(os.listdir(path))
        p = np.random.permutation(len(files))
        files2 = files[p]

        for j, file1 in enumerate(tqdm(files)):
            try:
                if check_paths(file1, dataset, args):
                    file2 = files2[j]
                    image1, detection_result1 = detect_landmarks(
                        path=os.path.join(path,file1)
                        ) 
                    image2, detection_result2 = detect_landmarks(
                        path=os.path.join(path,file2)
                        ) 
                    m1, m2 = crop_face(image1)

                    for name in args:
                        save_path = os.path.join(dataset,"p_"+name,str(i))
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        if name == "cheeks":
                            image3 = replace_feature(
                                "left_cheek",
                                image1,
                                image2,
                                detection_result1,
                                detection_result2
                            )

                            image = replace_feature(
                                "right_cheek",
                                image3,
                                image2,
                                detection_result1,
                                detection_result2
                            )

                        else:
                            image = replace_feature(
                                name,
                                image1,
                                image2,
                                detection_result1,
                                detection_result2
                            )

                        cropped_image = image[m1[1]:m2[1],m1[0]:m2[0]]
                        cv2.imwrite(os.path.join(save_path,file1), cropped_image)
            except:
                print(file1, file2)

def draw_boxes(path):
    image, detection_result = detect_landmarks(path=path)
    #m1, m2 = crop_face(image)

    for i in range(len(names2)):
        _, p1, p2 = crop_feature(names2[i], image, detection_result)
        image = cv2.rectangle(image, p1, p2, (255,255,255), 1)
        #cropped_image = image[m1[1]:m2[1],m1[0]:m2[0]]
    
    return image

if __name__ == "__main__":
    #generate_dataset("MEBeauty/images", "mediapipe")
    #generate_dataset("SCUT-FBP5500/images", "eyes", "eyebrows")
    #permutation("MEBeauty/images", 1, 6, *names2)
    import matplotlib.pyplot as plt
    cv2.imwrite("boxes.jpg",draw_boxes("MEBeauty/images/605.jpg"))

