import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from data.PhyreDataset import PhyreDataset
from visualization.rendering import SimplePhyreRenderer, PhyreTrajectoryRenderer

colors = [(0, 0, 256), (0, 256, 0), (256, 0, 0)]


def get_image_from_frame(frame):
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    for obj in frame:
        x = obj[0]
        y = obj[1]

        diameter = obj[4]
        is_blue = obj[5]
        is_green = obj[6]
        is_yellow = obj[7]
        idx = 0 if is_blue else 1 if is_green else 2 if is_yellow else 0
        cv2.circle(image, (int(255 * x), int(255 * y)), radius=int((255 * diameter) / 2), thickness=-1,
                   color=colors[idx])
    image = cv2.flip(image, 0)
    return image


def run_to_video(run):
    images = [get_image_from_frame(frame) for frame in run]

    video_writer = cv2.VideoWriter("../data/video_file.avi", cv2.VideoWriter_fourcc(*'mp4v'), 20, (256, 256))

    for image in images:
        video_writer.write(image)
    video_writer.release()
    return images


if __name__ == '__main__':
    dset = PhyreDataset(
        "../../../Development/phyre-proj/PHYRE-diffusion/phyrediff/data/images/phyre_diff_00_1_task_1_action_latents.h5")
    print(len(dset))
    for run in dset:
        print(len(run))
    first_run = dset[0]
    run = run_to_video(first_run)
    renderer = PhyreTrajectoryRenderer()
    renderer.composite("../data/run.png", first_run[np.newaxis, ...])
