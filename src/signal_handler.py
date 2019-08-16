import cv2
from global_config import frame_shape
from model_factory.toy_cnn_ae import autoencoder, encoder
import numpy as np
import os
from build_mvp_model import get_index_file, PROJECT_ROOT, download_file, video_url
from keras.models import load_model
from matplotlib import pyplot as plt
import sklearn
from sklearn import metrics
def get_frames():
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (1, 30)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    # download_all_videos()
    model_name = 'version_1.m5'
    encoder_name = 'version_1_encoder.m5'
    ae = load_model(os.path.join(PROJECT_ROOT, 'models', model_name))
    data_folder = os.path.join(PROJECT_ROOT, 'data')
    file_list = get_index_file(step_size=30)
    encoder = load_model(os.path.join(PROJECT_ROOT, 'models', encoder_name))
    q = [0] * 200
    prev_frame_embedding = None
    reference_frame = None
    for this_file_url in file_list:
        print(this_file_url)
        file_name = this_file_url.split('/')[-1]
        file_path = os.path.join(data_folder, file_name)
        # continue
        if not os.path.exists(file_path):
            download_file(video_url(this_file_url), file_path)

        cap = cv2.VideoCapture(file_path)
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        buf = []
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
                h, w = frame_shape
                # y_sample_idx = np.random.randint(0, frame.shape[0] - h, sample_batch_size)
                # x_sample_idx = np.random.randint(0, frame.shape[1] - w, sample_batch_size)
                # unioned = zip(y_sample_idx, x_sample_idx)
                # for coor in unioned:
                #     y, x = coor
                x, y = 180, 180
                cropped_frame = frame[y:y + h, x:x + w]
                ret = encoder.predict(np.array([cropped_frame]))[0]
                recon = ae.predict(np.array([cropped_frame]))[0].astype(np.uint8)
                counter += 1
                if prev_frame_embedding is not None:

                    diss = metrics.pairwise.euclidean_distances([ret], [prev_frame_embedding])
                    q.pop(0)
                    q.append(diss[0][0])

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                    cv2.putText(cropped_frame, str(diss[0][0]),
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType,
                                )
                    fig = plt.figure()
                    plot = fig.add_subplot(111)
                    plot.plot(q,)
                    fig.canvas.draw()

                    # Now we can save it to a numpy array.
                    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plt.cla();plt.clf();plt.close()
                    cv2.imshow('ts', data)
                    cv2.imshow('Live', cropped_frame)
                    cv2.imshow('Reconstruction', recon)
                    cv2.imshow('reference', reference_frame)
                if prev_frame_embedding is None:
                    # set first frame as reference
                    prev_frame_embedding = ret
                    reference_frame = cropped_frame

            # Break the loop
            else:
                break


if __name__ == "__main__":
    get_frames()