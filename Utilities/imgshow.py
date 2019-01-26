import gym
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np

env = gym.make('PongDeterministic-v4')
frame = env.reset()

for _ in range(1,1):
    frame,_,_,_ = env.step(1)

grayscaled = rgb2gray(frame) # 210x160
#grayscaled = grayscaled[16:201,:]
processed_observe = np.uint8(resize(grayscaled, (84, 84), mode='constant') * 255)

plt.imshow(processed_observe, cmap="gray")
plt.show()

