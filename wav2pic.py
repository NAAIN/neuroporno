from scipy.io import wavfile
from PIL import Image
import numpy as np
orig_rate,orig = wavfile.read("resources/wav_porn_files/orig.wav")
if orig.ndim == 2:
    print("Warning, Original file is Stereo. converting to mono..")
    orig = np.mean(orig, axis=1).astype(orig.int16)
orig = np.array(orig,dtype=int)
del wavfile
x,y,resizeTimes = (0,0,0)
print("Orig len:  ",len(orig))
pic = Image.new("RGB",(1000,1000))
w,h = pic.size

try:
    for i in range(len(orig)):
        if h == (y - 1):
            pic.resize((w,h + 100))
            resizeTimes += 1
        if w == (x - 1):
            pic.resize((w + 100,h))
            resizeTimes += 1
        w,h = pic.size
        if x != w:
            pic.putpixel((x,y),(0,0,int(np.interp(orig[i], [-32768,32767], [0,255]))))
            x += 1
        else:
            y += 1
            x = 0
except IndexError:
    print("ban")

pic.save("resources/wav_porn_files/a.png")
print(resizeTimes)