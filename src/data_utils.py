from download_utils import get_index_file


class VideoDatabaseAccess(object):
    def __init__(self):
        self.__raw_file_list__ = get_index_file(shuffle=False)
        self.__int_idx__ = [int(x.split('.')[0]) for x in self.__raw_file_list__]
        self.min = min(self.__int_idx__)
        self.max= max(self.__int_idx__)




if __name__ == "__main__":
    test = VideoDatabaseAccess()
