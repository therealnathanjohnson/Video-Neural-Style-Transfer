# Video Neural Style Transfer

The code to carry out the complete Video Style Transfer pipline, including selecting keyframes from a video a user wants to stylize, stylizing keyframes and propagating the styles from the keyframes to rest of the frames, is provided in `full_pipeline.py`. 

Note that when selecting keyframes, a user may want to run `display_frames` multiple times, with different configurations, to visually inspect each indvidual frame, before deciding which ones are most suitable to be keyframes. Further information on this is provided in `keyfame_selection_utils.py`.

Note: It may be necessary to clear the `data\keyframes` and `data\keyframes_stylized` folders if different keyframes are chosen


## Usage

### First Steps

Clone Repostiory
```
$ git clone https://github.com/therealnathanjohnson/Video-Neural-Style-Transfer.git
$ cd Video-Neural-Style-Transfer
```

Install Requirements
```
$ pip install -r requirements.txt
```

### Display Frames in a Folder to Decide on Keyframes

Extract all frames from the video and store in a folder so they can be visually inspected
```
$ python keyframe_selection_utils.py \
  --video "data/video.mp4" \
  --keyframes_dir "data/view_frames"
```

Extract only certain chosen frames and store in folder (frames 5 and 10, in this case)
```
$ python keyframe_selection_utils.py \
  --video "data/video.mp4" \
  --keyframes_dir "data/keyframes" \
  --chosen_keyframes 5 10
```

Extract all frames within the min and max chosen keyframes (frames 5 to 15, in this case)
```
$ python keyframe_selection_utils.py \
  --video "data/video.mp4" \
  --keyframes_dir "data/keyframes" \
  --chosen_keyframes 5 10 15 \
  --use_range
```

The latter two options allow you to inspect certain sections of the video without creating an image per frame, which could require large amounts of disk space.

### Video Neural Style Transfer Pipeline

This command will transfer the style from `style.png` to the video. The `chosen_keyframes` parameter should be set based on the previous, exploratory step. These should coincide with the most important frames of the video.

```
$ python full_pipeline.py \
  --video "data/video.mp4" \
  --style "data/style.png" \
  --keyframes_dir "data/keyframes" \
  --stylized_keyframes_dir "data/keyframes_stylized" \
  --output_dir "data/output" \
  --alpha 0.5 \
  --chosen_keyframes 0 19, 46, 84
```

The `--use_edges`, `--use_temporal_error_term`, `--no_optical_flow`, and `--backward_sweep` flags can be employed to use edge gaussian pyramids during PatchMatch, include the temporal error term during PatchMatch, not use optical flows, and include a backward sweep, after the forward sweep, respectively. 

Note: It may be necessary to empty the `keyframes_dir`, `stylized_keyframes_dir` and `output_dir`, before running this command.
