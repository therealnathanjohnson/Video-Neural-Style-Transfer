from pathlib import Path
import argparse
import numpy as np
import imageio

from keyframe_selection_utils import display_frames, save_selected
from neural_neighbour_style_transfer import complete_process
from video_style_transfer import Image_Analogies_Sweeps

if __name__ == "__main__":
    #get command line arguments
    parser = argparse.ArgumentParser(description="Video Neural Style Transfer Pipeline")

    parser.add_argument("--video", type=str, default="data/video.mp4",
                        help="Path to input video which must be stylized")
    
    parser.add_argument("--style", type=str, default="data/style.png",
                        help="Path to style image")
    
    parser.add_argument("--keyframes_dir", type=str, default="data/keyframes",
                        help="Directory to store extracted keyframes")
    
    parser.add_argument("--stylized_keyframes_dir", type=str, default="data/keyframes_stylized",
                        help="Directory to store stylized keyframes")
    
    parser.add_argument("--output_dir", type=str, default="data/output",
                        help="Directory for final output video")
    
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Stylization strength")

    #with `action="store_true"`, use_edges will be false unless this option is used
    parser.add_argument("--use_edges", action="store_true",
                    help="Use edge features in Image Analogies")

    parser.add_argument("--use_temporal_error_term", action="store_true",
                        help="Enable temporal consistency penalty")
    
    parser.add_argument("--no_optical_flow", action="store_true",
                        help="Disable optical flow")
    
    parser.add_argument("--backward_sweep", action="store_true",
                        help="Run PatchMatch backward sweep")

    #list chosen keyframe numbers
    parser.add_argument("--chosen_keyframes", type=int, nargs="+", required=True,
                    help="List of keyframe indices (e.g. --frames 0 19 46 84)")
    
    args = parser.parse_args()

    keyframe_path = Path(args.keyframes_dir)
    stylized_keyframe_path = Path(args.stylized_keyframes_dir)
    output_path = Path(args.output_dir)

    #create folders if they don't exist
    keyframe_path.mkdir(parents=True, exist_ok=True)
    stylized_keyframe_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    #convert back to strings
    keyframe_path = str(keyframe_path)
    stylized_keyframe_path = str(stylized_keyframe_path)
    output_path = str(output_path)
    
    style_img_path = args.style
    video_path = args.video #path of the video the user would like to stylize

    #extract the chosen keyframes from the video
    #REMEMBER: The first and last frame have to be chosen as keyframes to be compatible
    # with the Video NST function
    #To select the keyframes, you may need to call display_frames with different configurations
    # so you can visibily inspect each frame and decide which ones are most important
    # More information is provided in the `keyframe_selection_utils.py` file
    store, frame_ids = display_frames(
        video_path=video_path, 
        chosen_frames = args.chosen_keyframes, 
        use_range=False, 
        plt_show=False
    )
    #save the selected keyframes to a folder
    save_selected(store, frame_ids, folder_path=keyframe_path)
    
    #loop through all keyframes and run NNST to get stylized keyframes
    for entry in Path(keyframe_path).iterdir():
        if entry.is_file():
            #the number after the last underscore is the frame position
            frame_id = int(entry.stem.split("_")[-1])
            final_path = f"{stylized_keyframe_path}/frame_{frame_id}.png"
            #run NNST to get stylized version of keyframe
            complete_process(
                base_img_path = entry,
                style_img_path = style_img_path,
                final_path = final_path,
                alpha = args.alpha,
                lossless=False,
                reduce_memory_at_full_scale=True
            )
            print("style transferred keyframe", frame_id)

    #create sweeping Image Analogies instance. This is used for Video NST.
    ia = Image_Analogies_Sweeps(
        A_folder = keyframe_path, #folder for keyframes
        A_prime_folder = stylized_keyframe_path, #folder of stylized keyframes
        video_path = video_path, #video to stylize
        data_dir_path = output_path,
        use_edges = args.use_edges, 
        use_temporal_error_term = args.use_temporal_error_term, 
        use_optical_flow = not args.no_optical_flow
    )
    #run patch match sweep processing which will allow us to propagate styles from the keyframes, to the other frames
    nnf_list = ia.patch_match_sweep(backward_sweep=args.backward_sweep)
    #use NNFs to construct list of stylized frames
    output_list = ia.construct_video(nnf_list)
    #save stylized frames as video
    output_list_numpy = np.stack([x.cpu() for x in output_list])
    output_list_numpy = (output_list_numpy * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(ia.data_dir_path+"/final_output.mp4", np.permute_dims(output_list_numpy, (0,2,3,1)), fps=25)
    
