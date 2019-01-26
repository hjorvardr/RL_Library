import numpy as np
import gym
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.transform import resize

def plot_gray(observe):
    grayscaled = rgb2gray(observe) # 210x160
    grayscaled = grayscaled[16:201,:]
    processed_observe = np.uint8(resize(grayscaled, (84, 84), mode='constant') * 255)
    plt.imshow(processed_observe)
    plt.gray()
    plt.show()

env = gym.make("BreakoutNoFrameskip-v4")
env.reset()
for _ in range(5):
    frame,_,_,_ = env.step(3)

frame,_,_,_ = env.step(1)
for _ in range(5):
    frame,_,_,_ = env.step(2)

plot_gray(frame)

