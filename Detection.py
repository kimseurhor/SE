import time
import cv2
from PIL import Image
from facenet_pytorch import MTCNN

def detection(ip):
    mtcnn = MTCNN(image_size=2160, margin=20, keep_all=True, post_process=False)
    try:
        # video_capture = videoCapture(ip)
        video_capture = cv2.VideoCapture(ip)
        print("Accessed Camera :", ip)
    except:
        print("Cannot Accessed Camera ")

    cnt = 0
    while cnt < 5:

        # Grab a single frame from video
        ret, frame = video_capture.read()

        try:
            rgb_frame = frame[:, :, ::-1]

            # Convert frame to mtcnn frame which will use by facenet_pytorch
            mtcnn_frame = Image.fromarray(rgb_frame)

            # Declare and read directory which we will store cropped face images
            # pic_path = directory + "/" + str(cnt) + ".png"
            pic_path = "D:/classwork/SE/Project/face_Detect" + "/" + str(cnt) + ".png"

            # Save image name to queue
            img_list = mtcnn(mtcnn_frame, save_path=pic_path)
            print(img_list)
            cnt += 1
        except Exception as e:
            print("Error 38: ", e)

        time.sleep(0.5)

    print("----------------------------------")
    print("Stop Detection...")
    print("----------------------------------")


if __name__ == "__main__":
    directory = "D:/classwork/SE/Project/face_Detect"
    detection(0)