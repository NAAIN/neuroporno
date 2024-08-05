from scipy.io import wavfile
from PIL import Image
import numpy as np

pic = Image.open("a.png")
w,h = pic.size
x,y = (0,0)
samples = []
for i in range(w*h):
    if x != w:
        pixel = pic.getpixel((x,y))
        x += 1
    else:
        y += 1
        x = 0
    samples.append(np.interp(pixel,[0,255],[-1,1]))
wavfile.write("resources/wav_porn_files/pic2wav_out.wav",44100,np.array(samples))