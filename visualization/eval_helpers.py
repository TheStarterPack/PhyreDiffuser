import numpy as np
import torch
import phyre
import random

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

def get_color(one_hot_array):
    idx = np.argmax(one_hot_array)
    colors = {0: "red", 1: "lightgreen", 2: "blue", 3: "purple", 4: "grey", 5: "black"}
    return np.array(to_rgb(colors[idx]))

def paint_trajectory_simple(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    (seq_len, n_objs, n_props) = x.shape
    plt.figure()
    for n in range(n_objs):
        colors = get_color(x[0,n,5:11])[None,:]*np.linspace(0.2, 1, seq_len)[:,None]
        plt.scatter(x[:,n,0], x[:,n,1], s=x[0,n,4]*16000, alpha=0.1, c=colors)
        plt.xlim(0,1)
        plt.ylim(0,1)
    plt.show()
    plt.close()

def transform_to_phyre_features(x):
    x = np.concatenate([x[...,:2],
                        np.arctan2(x[...,2], x[...,3])[...,None]/(2*np.pi),
                        x[...,4:15]], axis=-1)
    x[...,:2] = np.clip(x[...,:2],0,1) # Clip positions
    return x
    
def paint_trajectory(x, create_image=True, return_frames=False, return_cum_image=False):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    (seq_len, n_objs, n_props) = x.shape
    
    # Transform x into feature shape expected by Phyre:
    x = transform_to_phyre_features(x)
    
    frames = []
    cum_img = None
    for l in range(seq_len):
        img = phyre.observations_to_uint8_rgb(phyre.featurized_objects_vector_to_raster(x[l]))/255
        frames.append(img)
        
        if create_image or return_cum_image:
            # Image composition by alpha blending
            rgba_img = np.dstack([img, np.where(img[:,:,0] + img[:,:,1] + img[:,:,2] == 3, 1e-10, 0.1)])
            if cum_img is None:
                cum_img = rgba_img
            cum_img_RGB = cum_img[...,:3]
            rgba_img_RGB = rgba_img[...,:3]
            cum_img_A = cum_img[...,3]
            rgba_img_A = rgba_img[...,3]
            outA = rgba_img_A + cum_img_A*(1-rgba_img_A)
            outRGB = (rgba_img_RGB*rgba_img_A[...,np.newaxis] + cum_img_RGB*cum_img_A[...,np.newaxis]*(1-rgba_img_A[...,np.newaxis])) / outA[...,np.newaxis]
            cum_img = np.dstack((outRGB,outA))
    
    if create_image:
        plt.figure()
        plt.imshow(cum_img)
        plt.show()
        
    if return_frames:
        return frames
    elif return_cum_image:
        return cum_img

   
def permute_objs(trajectory):
    out = trajectory.clone()
    for trajectory in range(len(out)):
        colors_n_shape = [out[trajectory,0,obj,5:15].clone() for obj in range(out.shape[-2])]
        random.shuffle(colors_n_shape)
        for obj in range(out.shape[-2]):
            out[trajectory, :, obj, 5:15] = colors_n_shape[obj]
    return out


def move_objects(img_latent):
    # Updates img_latent in place
    # Create figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)

    rect_size = 26 #(10% of image width)
    
    # Create an initial image array
    image = phyre.observations_to_uint8_rgb(phyre.featurized_objects_vector_to_raster(transform_to_phyre_features(img_latent)))/255
    
    # Display the initial image
    img = ax.imshow(image, origin='upper', extent=(0, 256, 0, 256))
    
    # Add rectangles
    rectangles = []
    text_objs = []
    for i, o in enumerate(list(img_latent)):
        # For each object
        rect = Rectangle(((o[0]-0.05)*256, (o[1]-0.05)*256), rect_size, rect_size, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        rectangles.append(rect)

        # Add text annotation for object index
        fontsize = 15
        text_obj = ax.text((o[0]-0.05)*256 + rect_size//2 - fontsize//2, (o[1]-0.05)*256 + rect_size//2 - fontsize//2, str(i), weight='bold',
                           fontsize=fontsize, color='yellow', path_effects=[pe.withStroke(linewidth=1, foreground="black")], alpha=0.5)
        text_objs.append(text_obj)
    
    # Make rectangles draggable
    interactive_objs = []
    for i, rect in enumerate(rectangles):
        dr = DraggableRectangle(rect, rect_size, text_objs[i], fontsize, i, ax, img, img_latent)
        dr.connect()
        interactive_objs.append(dr)
    
    plt.show()
    print("When you're done, don't forget to run 'stop_moving_objects(interactive_objs)'!")
    return interactive_objs

# Stop the processes that run in the background for the updating operation above.
def stop_moving_objects(interactive_objs):
    for rect in interactive_objs:
        if hasattr(rect, 'animation'):
            rect.animation.event_source.stop()
            print("Stopped animation.")
        rect.disconnect()
    plt.close()
    print("Successfully stopped all processes.")


class DraggableRectangle:
    def __init__(self, rect, rect_size, text_obj, fontsize, obj_id, ax, image, img_latent):
        self.rect = rect
        self.rect_size = rect_size
        self.text_obj = text_obj
        self.fontsize = fontsize
        self.ax = ax
        self.image = image
        self.obj_id = obj_id
        self.press = None
        self.delay = 10*len(img_latent)  # Delay in milliseconds, longer the more objects there are
        self.img_latent = img_latent

    def connect(self):
        self.cidpress = self.rect.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def disconnect(self):
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        contains, _ = self.rect.contains(event)
        if not contains:
            return
        self.press = (self.rect.get_x() - event.xdata, self.rect.get_y() - event.ydata)

    def on_motion(self, event):
        if self.press is None:
            return
        if event.inaxes != self.ax:
            return
        # Update hidden rectangles
        self.rect.set_x(event.xdata + self.press[0])
        self.rect.set_y(event.ydata + self.press[1])

        half_width = self.rect_size//2 + 1
        center_x = (event.xdata + self.press[0] + half_width)
        center_y = (event.ydata + self.press[1] + half_width)
        self.text_obj.set_position((center_x - self.fontsize//2, center_y - self.fontsize//2))

        # Also update actual img_latents
        self.img_latent[self.obj_id, 0] = center_x/256
        self.img_latent[self.obj_id, 1] = center_y/256

        if not hasattr(self, 'animation'):
            self.animation = FuncAnimation(self.rect.figure, self.update_image, interval=self.delay, repeat=False)
        self.rect.figure.canvas.draw_idle()

    def on_release(self, event):
        self.press = None

    def update_image(self, *args):
        image_data = phyre.observations_to_uint8_rgb(phyre.featurized_objects_vector_to_raster(transform_to_phyre_features(self.img_latent)))/255
        self.image.set_data(image_data)
        self.ax.figure.canvas.draw_idle()