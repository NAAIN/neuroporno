import numpy as np
import cv2

def seam_carve(image_path, new_width, new_height):
    def calculate_energy(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energy = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return energy

    def find_seam(energy):
        rows, cols = energy.shape
        seam = np.zeros(rows, dtype=np.uint32)
        cost = energy.copy()
        for i in range(1, rows):
            for j in range(cols):
                min_cost = cost[i-1, j]
                if j > 0:
                    min_cost = min(min_cost, cost[i-1, j-1])
                if j < cols - 1:
                    min_cost = min(min_cost, cost[i-1, j+1])
                cost[i, j] += min_cost
        seam[-1] = np.argmin(cost[-1])
        for i in range(rows-2, -1, -1):
            j = seam[i+1]
            if j > 0 and cost[i, j-1] < cost[i, j]:
                j -= 1
            if j < cols-1 and cost[i, j+1] < cost[i, j]:
                j += 1
            seam[i] = j
        return seam

    def remove_seam(img, seam):
        rows, cols, _ = img.shape
        output = np.zeros((rows, cols-1, 3), dtype=np.uint8)
        for i in range(rows):
            j = seam[i]
            output[i, :, 0] = np.delete(img[i, :, 0], j)
            output[i, :, 1] = np.delete(img[i, :, 1], j)
            output[i, :, 2] = np.delete(img[i, :, 2], j)
        return output

    img = cv2.imread(image_path)
    orig_height, orig_width = img.shape[:2]

    while orig_width > new_width:
        energy = calculate_energy(img)
        seam = find_seam(energy)
        img = remove_seam(img, seam)
        orig_width -= 1

    while orig_height > new_height:
        img = np.rot90(img, 1, (0, 1))
        energy = calculate_energy(img)
        seam = find_seam(energy)
        img = remove_seam(img, seam)
        img = np.rot90(img, -1, (0, 1))
        orig_height -= 1

    cv2.imwrite('resized_image.jpg', img)
    return img

image_path = 'path_to_your_image.jpg'
new_width = 300
new_height = 400
resized_image = seam_carve(image_path, new_width, new_height)
