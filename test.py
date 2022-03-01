import numpy as np


def rank_interaction(pixel):
    pixel.sort()
    pixel_list = pixel.tolist()

    pixel_dict = {}

    for i in range(len(pixel_list)):
        pixel_dict[i] = pixel_list[i]
    pixel_dict = np.array(list(pixel_dict.values()))
    return pixel_dict


print(rank_interaction(np.array([3, 5, 3, 1, 5, 4, 7, 8, 9, 10])))
