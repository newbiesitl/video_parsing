import os, random, cv2

from download_utils import get_index_file, download_file_given_file_name
from global_config import LABEL_DIR, FRAME_SIZE, ATTENTION_COOR, MIN_TS, DATA_DIR, FPS, FOOTAGE_LENGTH

import math



def get_label_file_index(label_file_name, step_size=1, shuffle=True):
    filename = os.path.join(LABEL_DIR, label_file_name)
    ret = [line.strip() for line in open(filename)][::step_size]
    if shuffle:
        random.shuffle(ret)
    return ret


def open_video(file_path, fps=1, h=FRAME_SIZE[0], w=FRAME_SIZE[1],
               y=ATTENTION_COOR[0], x=ATTENTION_COOR[1], normalize=True):
    if not os.path.exists(file_path):
        raise FileNotFoundError('file %s not found' % file_path)
    cap = cv2.VideoCapture(file_path)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        raise ValueError('Error opening video file %s' % file_path)
    frame_count = 0
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_count += 1
        if frame_count % fps != 0:
            continue
        if ret == True:
            normalize_term = 255 if normalize else 1
            # normalize frame value
            frame = frame.astype('float32') / normalize_term
            cropped_frame = frame[y:y + h, x:x + w]
            yield cropped_frame
        # Break the loop
        else:
            break


class __binary_search_indexing__(object):
    def __init__(self, values, window_size):
        '''
        This BS find the target that target is within [target, target+window size] range
        window size is video clip length
        :param values:
        :param window_size:
        '''
        values.sort()
        self.w = window_size
        self.values = values
        self.root_pos = len(values) // 2
        self.l = 0
        self.r = len(self.values)

    def __get_l_pos__(self, cur, l, r):
        return math.floor((l+cur)/2)

    def __get_r_pos__(self, cur,  l, r):
        return math.ceil((cur+r)/2)

    def __find__(self, t,):
        cur = self.root_pos
        return self.__find_impl__(t, cur, self.l, self.r)

    def __find_impl__(self, t, root_pos, l, r):
        if self.values[root_pos] <= t < self.values[root_pos] + self.w:
            return self.values[root_pos]
        elif self.values[root_pos] < t:
            next_pos = self.__get_r_pos__(root_pos, l, r)
            if next_pos >= r:
                return None
            # print(root_pos, next_pos, r)
            return self.__find_impl__(t, next_pos, root_pos, r)

        elif self.values[root_pos] > t:
            next_pos = self.__get_l_pos__(root_pos, l, r)
            if next_pos <= l:
                return None
            return self.__find_impl__(t, next_pos, l, root_pos)




class VideoDatabaseAccess(object):
    def __init__(self):
        self.__raw_file_list__ = get_index_file(shuffle=False)
        self.__int_idx__ = [int(x.split('.')[0]) for x in self.__raw_file_list__]
        '''
        The problem is to find the left neighbour that contains target,
        the window size is the video length, so far as i know it's a fix number 4 secs
        so the termination condition is to return the current node if current node + window size > target 
        '''

        self.__index_tool__ = __binary_search_indexing__(self.__int_idx__, window_size=FOOTAGE_LENGTH)

        self.min = min(self.__int_idx__)
        self.max = max(self.__int_idx__)
        # because the clip is every 4 secs, i can do mod to extract the file name

    def get_closest_file_stream_given_ts(self, ts, h=FRAME_SIZE[0], w=FRAME_SIZE[1],
                                         y=ATTENTION_COOR[0], x=ATTENTION_COOR[1]):
        if ts < self.min or ts > self.max:
            raise ValueError("given time stamp %d outside range (%d, %d)" % (ts, self.min, self.max))
        ret = self.__index_tool__.__find__(ts)
        if ret is None:
            raise ValueError('streaming at time stamp %d not available' % (ts))
        file_name = str(ret)+'.ts'
        file_path = os.path.join(DATA_DIR, file_name)
        print('opening %s for timestamp %d' % (file_path, ts))
        if not os.path.exists(file_path):
            print('downloading file %s to %s...' % (file_name, file_path), end='')
            download_file_given_file_name(file_name)
            print('done.')
        return open_video(file_path, h=h, w=w, x=x, y=y), ret

    def get_frame_given_ts(self, ts, h=FRAME_SIZE[0], w=FRAME_SIZE[1],
                           y=ATTENTION_COOR[0], x=ATTENTION_COOR[1], get_left_most_file_ts=False):
        '''
        Timestamp in second
        :param ts: integer - second
        :return:
        '''
        stream, leftmost_exist_boundary = self.get_closest_file_stream_given_ts(ts, w=w, h=h, x=x, y=y)
        frame_counter = 0
        for frame in stream:
            cur_pos = frame_counter // FPS + leftmost_exist_boundary
            if ts == cur_pos:
                if get_left_most_file_ts:
                    return frame, leftmost_exist_boundary
                return frame

            frame_counter += 1


if __name__ == "__main__":
    target_ts = 1538076226
    test = VideoDatabaseAccess()
    f = test.get_frame_given_ts(target_ts)
    cv2.imshow('target frame %d' % target_ts, f)
    cv2.waitKey(0)