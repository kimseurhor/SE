import concurrent.futures
import os
import threading
from queue import Queue

import joblib
from PIL import Image

from Detection import detection
from preprocessing import ExifOrientationNormalize


def remove_student(label):
    idx_class = model_data.idx_to_class
    for key, val in idx_class.items():
        if val == label:
            del idx_class[key]
            # print(key)
            break
    model_data.idx_to_class = idx_class


def recognition(pic, visited, result):
    preprocess = ExifOrientationNormalize()

    img = Image.open(pic)

    # recognize image
    img = preprocess(img)

    faces = model_data(img)
    if faces:
        face = faces[0]

        try:
            if face.top_prediction.confidence > 0.7:
                result_label = face.top_prediction.label
                student_name, student_id = result_label.split("_")
                print("Pic :", pic, "Name: ", student_name,
                      face.top_prediction.confidence)
                if student_id in visited:
                    result.add(student_name)
                    # remove_student(result_label)
                else:
                    visited.add(student_id)
        except:
            pass



def process_image_files():
    visited = set()
    result = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num,
                                               thread_name_prefix='recogniser') as executor:
        futures = []
        for img in os.listdir(directory):

            image_file = directory + "/" + img
            # Submit a new process for each image file
            future = executor.submit(recognition, image_file, visited, result)
            futures.append(future)
        # Wait for all processes to complete
        concurrent.futures.wait(futures)



if __name__ == "__main__":
    directory = "D:/classwork/SE/Project/face_Detect"
    # detection(0)
    # MODEL_PATH = backend_path + "dataset/model/SE_B10_A.pkl"
    MODEL_PATH = "D:/classwork/SE/Project/Dataset/SE_B10_A.pkl"
    model_data = joblib.load(MODEL_PATH)
    worker_num = 1
    queue = Queue()
    t2 = threading.Thread(target=detection, args=(0,))
    t1 = threading.Thread(target= process_image_files, args=())
    t1.start()
    t2.start()
    t1.join()
    t2.join()