from basic_pitch.inference import predict_and_save
import glob
import os


if __name__ == '__main__':
    main_dataset_folder=os.path.join(os.path.expanduser("~"), "PaganiniOp1")
    model_name = 'basic_pitch'
    
    nos = sorted(glob.glob(os.path.join(main_dataset_folder, 'audio', '*')))
    for no in nos:
        audio_paths = sorted(glob.glob(os.path.join(no, '*.wav')))
        split = no.split('/')
        output_dir = '/'.join(split[:-2] + ['pitch', model_name] + split[-1:])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        predict_and_save(audio_paths, output_dir, save_midi=True, sonify_midi=False, save_model_outputs=True, save_notes=True)
