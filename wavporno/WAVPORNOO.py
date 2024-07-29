#Скриптяга который берёт семплы с одного файла и суёт их в другой перед этим сравнивая
#Вдохновился от чела который "Bad Apple!! But Each Frame (And Audio) is Made of Steamboat Willie"
#made with ❤️ by @NAAIN_heh
#крутилочки \/
diff_criterion = 10000 #критерий тупого сравнения "разницы" (если слишком много - будут огрызки,если слишком мало - ничего не изменится)
method = 1 #метод дрочильни семплов

output = []
mod_samples,orig_samples,diff = 0,0,0
if method > 3 or method < 1: exit(f"Unexpected method {method}")
print(f"diff_criterion = {diff_criterion}, Method = {method} (1 - compare with diff, 2 - compare with tolerance, 3 - quick sort), if incorrect - edit script")

from scipy.io import wavfile
import numpy as np
import time
start_time = time.time()

orig_rate,orig = wavfile.read("orig.wav")
if orig.ndim == 2:
    print("Warning, Original file is Stereo. converting to mono..")
    orig = np.mean(orig, axis=1).astype(orig.dtype)
orig = np.array(orig,dtype=float)

if method < 3:
    target_rate,target = wavfile.read("target.wav")
    if target.ndim == 2:
        print("Warning, Target file is Stereo. converting to mono..")
        target = np.mean(target, axis=1).astype(target.dtype)
    target = np.array(target,dtype=float)
    if orig_rate != target_rate: print("Warning, Sample rates of files dont math, output may be speeded or slowed up")
    if len(orig) > len(target): print("Warning, Original length more than target. processing may be slow")
    print("Target len:",len(target))

print("Orig len:  ",len(orig))

if method == 1:
    if len(orig) > len(target):
     from scipy.interpolate import interp1d
     m = interp1d([0,len(orig)],[0,len(target)])
    for i in range(len(orig)):
        for ii in range(len(target)):
            if len(orig) > len(target): diff = abs(orig[i] - target[int(m(i))]) 
            else: diff = abs(orig[i] - target[i])
            if diff < diff_criterion:
                output.append(target[ii])
                mod_samples+=1
                break
            else:
                output.append(orig[i])
                orig_samples+=1
                break
            
if method == 2:
    def compare_with_tolerance(num1, num2, tolerance):
     return abs(num1 - num2) <= tolerance
    for i in range(len(orig)):
     for ii in range(len(target)):
         if compare_with_tolerance(target[ii],orig[i],diff_criterion):
             output.append(target[ii])
             mod_samples+=1
             break
         else:
             output.append(orig[i])
             orig_samples+=1
             break

if method == 3:
    def partition(array, low, high):
        pivot = array[high]
        i = low - 1
        for j in range(low, high):
            if array[j] <= pivot:
                i = i + 1
                (array[i], array[j]) = (array[j], array[i])
        (array[i + 1], array[high]) = (array[high], array[i + 1])
        return i + 1
    def quickSort(array, low, high):
        if low < high:
            pi = partition(array, low, high)
            quickSort(array, low, pi - 1)
            quickSort(array, pi + 1, high)
    output = orig
    mod_samples = 1
    quickSort(output, 0, len(orig) - 1)

orig = []
target = []

while np.max(np.abs(output)) < 0.85:
    for i in range(len(output)):
        output[i]*= 1.3
    print(f"Max sample volume is {np.max(np.abs(output))}", end='\r')

while np.max(np.abs(output)) > 0.85:
    for i in range(len(output)):
        output[i]/= 1.3
    print(f"Max sample volume is {np.max(np.abs(output))}", end='\r')

if method < 3:
    modified_percent = (mod_samples / (mod_samples + orig_samples)) * 100
    original_percent = (orig_samples / (mod_samples + orig_samples)) * 100
    print(f"Modified samples:{mod_samples},Original samples:{orig_samples}")
    print(f"{modified_percent:.1f}% modified samples, {original_percent:.1f}% original samples")
print(f"Executed in {time.time() - start_time} seconds")
wavfile.write(f"out_method{method}.wav",orig_rate,np.array(output))