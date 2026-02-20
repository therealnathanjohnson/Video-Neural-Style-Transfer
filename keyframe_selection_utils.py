import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import time

def display_frames(
    video_path,
    chosen_frames = [],
    use_range = False,
    plt_show=True
):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened()

    #keep track of current frame
    frame_count = 0
        
    #store frames
    store = []
    
    #loop through all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV reads in BGR, convert to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #if chosen_frames have been provided, we check if the frame is in chosen_frames before showing
        #if chosen_frames have been provided, but use_range is chosen, we check if the frame is in the range of chosen_frames
        #if no chosen_frames have bene provided, we show all frames
        if ((chosen_frames != [] and frame_count in chosen_frames) or 
            (use_range == True and chosen_frames != [] and 
             frame_count >= min(chosen_frames) and frame_count<= max(chosen_frames)) or 
            (chosen_frames == [])):
            store.append(frame_rgb)
            if plt_show:
                plt.figure(figsize=(6, 3))
                plt.imshow(frame_rgb)
                plt.axis("off")
                plt.title(f"Frame {frame_count}")
                plt.show()

        frame_count += 1

    cap.release()
    
    #we track the frame_ids since it is useful when saving the images
    if chosen_frames != []:
        #if frames were selected, and use_range is false, chosen_frames is our frame_id list
        #If use_range is selected the frames_ids should list all frames in the range of chosen_frames
        frame_ids = chosen_frames if use_range == False else [i for i in range(min(chosen_frames), max(chosen_frames)+1)]
    else:
        #if no frames were selected. frame_ids is the list ranging from 0 to the number of frames in the video
        frame_ids = [i for i in range(frame_count)]
        
    return np.stack(store), frame_ids

def save_selected(store, frame_ids, folder_path):
    #we should have a frame_id per image
    assert store.shape[0] == len(frame_ids)
    #loop through all images and save
    for i in range(store.shape[0]):
        Image.fromarray(store[i]).save(f"{folder_path}/frame_{frame_ids[i]}.png")

if __name__ == "__main__":
    #Before calling our video style transfer method, we need to first select keyframes (which are then stylized using NNST).
    #To do this, one can use display_frames to see every single frame before selecting keyframes

    #Calling `display_frames` with `chosen_frames` set to [] displays every frame of the video
    #Calling `display_frames` with `chosen_frames` set to, for example, [5, 10, 15], displays these select frames
    #If `use_range` is also set to True, ever frame between 5 and 15, exlusive, will be displayed

    #In a Jupyter Notebook, the displayed images will be shown inside a cell. In other cases, an external window will be opened for
    # each displayed frame, in which case it may be easier to set `plt_show` to false, and then use `save_selected` to save the frames
    # to a folder where they can easily be viewed individually.

    #Regardless, once frames have been selected, `save_selected` must be used and the frames must be stored in a folder so later programs
    # can access them

    store, frame_ids = display_frames(
        video_path="data\video.mp4", 
        chosen_frames = [5, 10, 15], 
        use_range=True, 
        plt_show=True
    )

    save_selected(store, frame_ids, folder_path="data/keyframes")
