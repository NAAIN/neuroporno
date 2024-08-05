from scipy.io import wavfile
from PIL import Image
import numpy as np
orig_rate,orig = wavfile.read("resources/wav_porn_files/a.wav")
if orig.ndim == 2:
    print("Warning, Original file is Stereo. converting to mono..")
    orig = np.mean(orig, axis=1).astype(orig.dtype)
orig = np.array(orig,dtype=int)
del wavfile
x,y,resizeTimes = (0,0,0)
print("Orig len:  ",len(orig))
pic = Image.new("RGB",(1000,int(len(orig)/1000)))
w,h = pic.size

try:
    for i in range(len(orig)):
        if x != w:
            pic.putpixel((x,y),(0,0,int(np.interp(orig[i], [-32768,32767], [0,255]))))
            x += 1
        else:
            y += 1
            x = 0
except (IndexError, KeyboardInterrupt):
    print("ban")

pic.save("resources/wav_porn_files/a.png")
print(resizeTimes)