from pathlib import Path
from neural_neighbour_style_transfer import complete_process
from video_style_transfer import Image_Analogies_Sweeps

if __name__ == "__main__":

    keyframe_path = 'data/keyframes'
    stylized_keyframe_path = 'data/keyframes_stylized'
    style_img_path = 'data/style.png'
    
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
                alpha = 0.5,
                lossless=False,
                reduce_memory_at_full_scale=True
            )

    #create sweeping Image Analogies instance. This is used for Video NST.
    ia = Image_Analogies_Sweeps(
        A_folder = keyframe_path, #folder for keyframes
        A_prime_folder = stylized_keyframe_path, #folder of stylized keyframes
        video_path = "data/video.mp4", #video to stylize
        data_dir_path = "data/output",
        use_edges = False, 
        use_temporal_error_term = False, 
        use_optical_flow = True
    )
    #run patch match sweep processing
    nnf_list = ia.patch_match_sweep(backward_sweep=False)
    #use NNFs to construct list of stylized frames
    output_list = ia.construct_video(nnf_list)
    #save stylized frames as video
    output_list_numpy = np.stack([x.cpu() for x in output_list])
    output_list_numpy = (output_list_numpy * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(ia.data_dir_path+"/final_output.mp4", np.permute_dims(output_list_numpy, (0,2,3,1)), fps=25)
    
