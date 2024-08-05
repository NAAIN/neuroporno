import cv2
import PIL.Image
import numpy as np
import io

def process_frame(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    jpeg_image = buffer.getvalue()
    return jpeg_image

def main():
    video_path = 'resources/video_shakal_files/a.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    
    if not ret:
        print("Cant read first frame, maybe file dont exists")
        return
    
    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    pil_image = PIL.Image.fromarray(frame_rgb)
    frame_size = pil_image.size
    out = cv2.VideoWriter('resources/video_shakal_files/a_out.mp4', fourcc,cap.get(cv2.CAP_PROP_FPS), frame_size)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(frame_rgb)
        jpeg_image = process_frame(pil_image, 0)
        jpeg_buffer = io.BytesIO(jpeg_image)
        jpeg_pil_image = PIL.Image.open(jpeg_buffer)
        frame_cv = np.array(jpeg_pil_image)
        frame_cv = cv2.cvtColor(frame_cv, cv2.COLOR_RGB2BGR)

        out.write(frame_cv)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()
