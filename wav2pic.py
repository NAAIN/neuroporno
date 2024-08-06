from scipy.io import wavfile
from PIL import Image
import numpy as np
orig_rate,orig = wavfile.read("resources/wav_porn_files/a.wav") #по факту там его не будет(он тажёлый до ужаса)
stereo = False
if orig.ndim == 2: stereo = True
#     print("Warning, Original file is Stereo. converting to mono..")
#     orig = np.mean(orig, axis=1).astype(orig.dtype)
orig = np.array(orig,dtype=int)
del wavfile
x,y,resizeTimes = (0,0,0)
print("Orig len:  ",len(orig))
pic = Image.new("RGB",((int(len(orig)/3000)),int(len(orig)/3700)),(127,127,127))
w,h = pic.size

if stereo:
    try:
        for i in range(len(orig)):
            if x != w:
                pic.putpixel((x,y),(0,int(np.interp(orig[i,0], [-32767,32767], [0,255])),int(np.interp(orig[i,1], [-32767,32767], [0,255]))))
                x += 1
            else:
                y += 1
                x = 0
    except (IndexError, KeyboardInterrupt):
        print("ban")
else:
    try:
        for i in range(len(orig)):
            if x != w:
                pic.putpixel((x,y),(0,0,int(np.interp(orig[i], [-32767,32767], [0,255]))))
                x += 1
            else:
                y += 1
                x = 0
    except (IndexError, KeyboardInterrupt):
        print("ban")

pic.save("resources/wav_porn_files/a.png")