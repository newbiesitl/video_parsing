import os

from download_utils import download_file, get_index_file, get_video_url

if __name__ == "__main__":
    # download_all_videos()
    import cv2
    from global_config import FRAME_SIZE, PROJECT_ROOT
    from model_factory.toy_cnn_ae import autoencoder, encoder
    import numpy as np
    from keras.models import load_model
    model_name = 'version_1.m5'
    encoder_name = 'version_1_encoder.m5'
    sample_batch_size = 50
    data_folder = os.path.join(PROJECT_ROOT, 'data')
    file_list = get_index_file(step_size=10)
    encoder_path = os.path.join(PROJECT_ROOT, 'models', encoder_name)
    model_path = os.path.join(PROJECT_ROOT, 'models', model_name)

    noise_factor = 0.3

    if os.path.exists(model_path):
        autoencoder = load_model(model_path)
    for this_file_url in file_list:
        print(this_file_url)
        file_name = this_file_url.split('/')[-1]
        file_path = os.path.join(data_folder, file_name)
        # continue
        if not os.path.exists(file_path):
            download_file(get_video_url(this_file_url), file_path)

        cap = cv2.VideoCapture(file_path)
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        buf = []
        noise_buf = []
        frame_count = 0
        fps = 4
        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            counter = 0
            frame_count += 1
            if frame_count % fps != 0:
                continue
            if ret == True:
                # normalize frame value
                frame = frame.astype('float32') / 255
                h, w = FRAME_SIZE
                y_sample_idx = np.random.randint(0, frame.shape[0]-h, sample_batch_size)
                x_sample_idx = np.random.randint(0, frame.shape[1]-w, sample_batch_size)
                unioned = zip(y_sample_idx, x_sample_idx)
                # unioned = [attention_coor]
                for coor in unioned:
                    y, x = coor
                    # x, y = 180, 180
                    cropped_frame = frame[y:y+h, x:x+w]
                    # add noise hmm
                    noise_frame = cropped_frame + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=cropped_frame.shape)
                    # recon = (autoencoder.predict(np.array([noise_frame]))[0])
                    # Display the resulting frame
                    buf.append(cropped_frame)
                    noise_buf.append(noise_frame)
                    counter += 1
                    # print(counter)
                    if counter == sample_batch_size:
                        break
                    # Press Q on keyboard to  exit
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                    # print(cropped_frame)
                    # print(noise_frame)
                    # print(recon)

                    # cv2.imshow('Frame_focus', cropped_frame)
                    # cv2.imshow('Model input', noise_frame)
                    # cv2.imshow('noise reconstruction', recon)
                    # clean_input = (autoencoder.predict(np.array([cropped_frame]))[0])
                    # cv2.imshow('noise reconstruction clean input', clean_input)
                    # cv2.imshow('Frame', frame)

            # Break the loop
            else:
                break
        buf = np.array(buf)
        noise_buf = np.array(noise_buf)
        autoencoder.fit(noise_buf, buf, epochs=10, verbose=2, batch_size=32)
        autoencoder.save(model_path)
        Encoder.set_weights(autoencoder.get_weights())
        Encoder.save(encoder_path)