import cv2
import numpy as np
from matplotlib import pyplot as plt

from visualization.eval_helpers import paint_trajectory

colors = [(0, 0, 256), (0, 256, 0), (256, 0, 0)]


class SimplePhyreRenderer:

    def __init__(self):
        pass

    def composite(self, savepath, observations):
        print("rendering images")
        images = []
        for i, run in enumerate(observations):
            image = np.full((256, 256, 3), 255, dtype=np.uint8)
            for frame in run:
                x_1, y_1, x_2, y_2, x_3, y_3 = frame[0], frame[1], frame[2], frame[3], frame[4], frame[5]
                cv2.circle(image, (int(x_1), int(y_1)), radius=1, color=colors[0], thickness=-1)
                cv2.circle(image, (int(x_2), int(y_2)), radius=1, color=colors[1], thickness=-1)
                cv2.circle(image, (int(x_3), int(y_3)), radius=1, color=colors[2], thickness=-1)
            images.append(image)
            images.append(np.zeros((256, 10, 3), dtype=np.uint8))
        final_image = np.concatenate(images, axis=1)
        final_image = cv2.flip(final_image, 0)
        cv2.imwrite(f"{savepath}", final_image)


class PhyreTrajectoryRenderer:
    def __init__(self):
        pass

    def composite(self, savepath, observations):
        images = []
        for idx, observation in enumerate(observations):
            reshaped_observations = np.reshape(observation, (-1, 3, 15))
            image = paint_trajectory(reshaped_observations, create_image=False, return_frames=False,
                                     return_cum_image=True)
            images.append(image)
            if idx != len(observations)-1:
                padding = np.zeros((256, 10, 4), dtype=np.uint8)
                images.append(padding)
        final_image = np.concatenate(images, axis=1)
        plt.imsave(f"{savepath}", final_image)
