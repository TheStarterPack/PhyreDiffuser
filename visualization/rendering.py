import cv2
import numpy as np

colors = [(0, 0, 256), (0, 256, 0), (256, 0, 0)]


class PhyreRenderer:

    def __init__(self):
        pass

    def composite(self, savepath, observations):
        print("rendering images")
        images = []
        for i, run in enumerate(observations):
            image = np.full((256, 256, 3), 255, dtype=np.uint8)
            for frame in run:
                x_1, y_1, x_2, y_2, x_3, y_3 = frame[0] * 255, frame[1] * 255, frame[2] * 255, frame[3] * 255, frame[
                    4] * 255, frame[5] * 255
                cv2.circle(image, (int(x_1), int(y_1)), radius=1, color=colors[0], thickness=-1)
                cv2.circle(image, (int(x_2), int(y_2)), radius=1, color=colors[1], thickness=-1)
                cv2.circle(image, (int(x_3), int(y_3)), radius=1, color=colors[2], thickness=-1)
            images.append(image)
            images.append(np.zeros((256, 10, 3), dtype=np.uint8))
        final_image = np.concatenate(images, axis=1)
        final_image = cv2.flip(final_image, 0)
        cv2.imwrite(f"{savepath}", final_image)
