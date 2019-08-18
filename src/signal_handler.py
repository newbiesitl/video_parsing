import cv2
from global_config import FRAME_SIZE, PROJECT_ROOT, EncoderDecoder, Encoder
import numpy as np
import os
from download_utils import download_file, get_index_file, get_video_url
from matplotlib import pyplot as plt
import matplotlib
from model_factory.knn_object_detection import build_knn

matplotlib.use('TkAgg')
def get_frames():
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (1, 30)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    # download_all_videos()
    data_folder = os.path.join(PROJECT_ROOT, 'data')
    file_list = get_index_file(step_size=50, shuffle=True)
    class_1_q = [0] * 200
    class_2_q = [0] * 200
    prev_frame_embedding = None
    knn = build_knn()
    for file_name in file_list:
        # file_name = this_file_url.split('/')[-1]
        file_path = os.path.join(data_folder, file_name)
        # continue
        if not os.path.exists(file_path):
            download_file(get_video_url(file_name), file_path)

        cap = cv2.VideoCapture(file_path)
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        frame_count = 0
        fps = 24
        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            counter = 0
            frame_count += 1
            if frame_count % fps != 0:
                continue
            if ret == True:
                frame = frame.astype('float32') / 255
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img2 = np.zeros_like(frame)
                img2[:, :, 0] = gray
                img2[:, :, 1] = gray
                img2[:, :, 2] = gray
                frame = img2
                h, w = FRAME_SIZE
                # y_sample_idx = np.random.randint(0, frame.shape[0] - h, sample_batch_size)
                # x_sample_idx = np.random.randint(0, frame.shape[1] - w, sample_batch_size)
                # unioned = zip(y_sample_idx, x_sample_idx)
                # for coor in unioned:
                #     y, x = coor
                x, y = 180, 180
                cropped_frame = frame[y:y + h, x:x + w]
                ret = Encoder.predict(np.array([cropped_frame]))[0]
                pred = knn.predict([ret])[0].split('_')[0]
                prob = knn.predict_proba([ret])[0]
                print(file_name, pred)
                recon = EncoderDecoder.predict(np.array([cropped_frame]))[0]
                counter += 1
                if prev_frame_embedding is not None:

                    class_1_q.pop(0)
                    class_2_q.pop(0)
                    class_1_q.append(prob[0])
                    class_2_q.append(prob[1])
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                    cv2.putText(cropped_frame, pred,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType,
                                )
                    fig = plt.figure()
                    plot = fig.add_subplot(111)
                    plot.plot(class_1_q,)
                    plot.plot(class_2_q,)
                    fig.canvas.draw()

                    # Now we can save it to a numpy array.
                    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plt.cla();plt.clf();plt.close()
                    cv2.imshow('ts', data)
                    cv2.imshow('Live', cropped_frame)
                    cv2.imshow('Reconstruction', recon)
                if prev_frame_embedding is None:
                    # set first frame as reference
                    prev_frame_embedding = ret

            # Break the loop
            else:
                break


if __name__ == "__main__":
    get_frames()