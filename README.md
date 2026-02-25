# Video Neural Style Transfer

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

Our Video NST algorithm relies on these keyframes to accurately and confidently style other frames, so it is crucial that the chosen keyframes have a reasonable coverage of the different objects and backgrounds in the video. If an object appears in the frames but not in the keyframes, our algorithm may not be able to style it well.

### Video Neural Style Transfer Pipeline

This command will transfer the style from `style.png` to the video. It extracts the `chosen_keyframes` from the video, stylizes then using NNST and propagates the style from the stylized keyframes to the rest of the frames. The `chosen_keyframes` parameter should be set based on the previous, exploratory step. These should coincide with the most important frames of the video.

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

### Bonus: Individual Sub-Module Commands

Neural Style Transfer (outputs at 224p, for the sake of speed):
```
$ python neural_style_transfer.py \
  --content "data/simba.jpg" \
  --style "data/style.png" \
  --output "data/output.png"
```
<i> Note: The above is NOT used in our video style transfer pipeline </i>

Neural Neighbour Style Transfer:
```
$ python neural_neighbour_style_transfer.py \
  --content "data/simba.jpg" \
  --style "data/style.png" \
  --output "data/output.png" \
  --alpha 0.5
```

Video Style Transfer (requires pre-populated `keyframes` and `stylized_keyframes` folders):
```
$ python video_style_transfer.py \
  --video "data/video.mp4" \
  --keyframes_dir "data/keyframes" \
  --stylized_keyframes_dir "data/keyframes_stylized" \
  --output_dir "data/output"
```
The `--use_edges`, `--use_temporal_error_term`, `--no_optical_flow`, and `--backward_sweep` flags are available for this command too. 
