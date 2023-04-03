import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
class Animation:
    def __init__(self, output_dir=None):
        '''
        function: show images in time series as animation
        input:
            img_time: [time, x, y]
            flag_save:bool
        '''
        self.output_dir = output_dir

    def show(self, img_time, title="test"):
        self.img_time = img_time
        fig = plt.figure()
        plt.title(title)
        self.ax = plt.imshow(np.abs(img_time[0]), cmap='gray')
        ani = FuncAnimation(
            fig, self.animate, init_func=self.init_for_animation, frames=img_time.shape[0],
            interval=150, blit=True)
        plt.show()

    def init_for_animation(self):
        '''Initialize ax data.'''
        self.ax.set_array(np.abs(self.img_time[0]))
        return (self.ax,)

    def animate(self, frame):
        '''Update frame.'''
        self.ax.set_array(np.abs(self.img_time[frame ]))
        return (self.ax,)

    def video_builder(self, img_time, title="test"):
        '''
            img_time: [time, x, y]
        '''
        file_name = title + '.avi'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        saving_path = os.path.join(self.output_dir, file_name)

        img_time = np.abs(img_time)
        size = (img_time.shape[-1], img_time.shape[-2])  #此处先Y在X
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(saving_path, fourcc, fps=10,
                                frameSize=size)
        for t in range(img_time.shape[0]):
            img = img_time[t]
            img_range = np.max(img) - np.min(img)
            if img_range > 0:
                img = (img - np.min(img)) / img_range * 255
            else:
                img = (img - np.min(img)) * 255
            img = np.expand_dims(img, -1).repeat(3, axis=-1)
            img = img.astype(np.uint8)
            img = cv2.resize(img, size)
            video.write(img)
        video.release()

    def save_data_pkl(self, data, title="test"):
        file_name = title + '.pkl'
        saving_path = os.path.join(self.output_dir, file_name)
        with open(saving_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)




