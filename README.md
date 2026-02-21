# Video Neural Style Transfer

The code to carry out the complete Video Style Transfer pipline, including selecting keyframes from a video a user wants to stylize, stylizing keyframes and propagating the styles from the keyframes to rest of the frames, is provided in `full_pipeline.py`. Note that when selecting keyframes, a user may want to run `display_frames` multiple times, with different configurations, to visually inspect each indvidual frame, before deciding which ones are most suitable to be keyframes. Further information on this is provided in `keyfame_selection_utils.py`.

