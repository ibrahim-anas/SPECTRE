# -*- coding: utf-8 -*-

import os, sys
import argparse
import torch
import numpy as np
import cv2
from skimage.transform import estimate_transform, warp
from tqdm import tqdm
from datasets.data_utils import landmarks_interpolate
from src.spectre import SPECTRE
from config import cfg as spectre_cfg
from src.utils.util import tensor2video
import torchvision
import fractions
import librosa
# from moviepy.editor import AudioFileClip
from scipy.io import wavfile
import collections
import gc

def extract_frames(video_path, detect_landmarks=True):
    videofolder = os.path.splitext(video_path)[0]
    os.makedirs(videofolder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)

    if detect_landmarks:
        from external.Visual_Speech_Recognition_for_Multiple_Languages.tracker.face_tracker import FaceTracker
        from external.Visual_Speech_Recognition_for_Multiple_Languages.tracker.utils import get_landmarks
        face_tracker = FaceTracker()

    imagepath_list = []
    count = 0
    face_info = collections.defaultdict(list)
    fps = fractions.Fraction(vidcap.get(cv2.CAP_PROP_FPS))

    with tqdm(total=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while True:
            success, image = vidcap.read()
            if not success:
                break

            if detect_landmarks:
                detected_faces = face_tracker.face_detector(image, rgb=False)
                landmarks, scores = face_tracker.landmark_detector(image, detected_faces, rgb=False)
                face_info['bbox'].append(detected_faces)
                face_info['landmarks'].append(landmarks)
                face_info['landmarks_scores'].append(scores)

            imagepath = os.path.join(videofolder, f'{count:06d}.jpg')
            cv2.imwrite(imagepath, image)  # save frame as JPEG file
            count += 1
            imagepath_list.append(imagepath)
            pbar.update(1)
            pbar.set_description("Preprocessing frame %d" % count)

    landmarks = get_landmarks(face_info)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list, landmarks, videofolder, fps

def crop_face(frame, landmarks, scale=1.0):
    image_size = 224
    left, right = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    top, bottom = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)

    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2],
                        [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    return tform

def main(args):
    print(f"Using device: {args.device}")  # Debug statement to confirm the device
    
    args.crop_face = True
    spectre_cfg.pretrained_modelpath = "pretrained/spectre_model.tar"
    spectre_cfg.model.use_tex = False

    spectre = SPECTRE(spectre_cfg, args.device)
    spectre.eval()

    image_paths, landmarks, videofolder, fps = extract_frames(args.input, detect_landmarks=args.crop_face)
    if args.crop_face:
        landmarks = landmarks_interpolate(landmarks)
        if landmarks is None:
            print('No faces detected in input {}'.format(args.input))
            return

    original_video_length = len(image_paths)
    image_paths.insert(0, image_paths[0])
    image_paths.insert(0, image_paths[0])
    image_paths.append(image_paths[-1])
    image_paths.append(image_paths[-1])

    landmarks.insert(0, landmarks[0])
    landmarks.insert(0, landmarks[0])
    landmarks.append(landmarks[-1])
    landmarks.append(landmarks[-1])

    landmarks = np.array(landmarks)
    L = 50
    indices = list(range(len(image_paths)))
    overlapping_indices = [indices[i: i + L] for i in range(0, len(indices), L - 4)]

    if len(overlapping_indices[-1]) < 5:
        overlapping_indices[-2] = overlapping_indices[-2] + overlapping_indices[-1]
        overlapping_indices[-2] = np.unique(overlapping_indices[-2]).tolist()
        overlapping_indices = overlapping_indices[:-1]

    overlapping_indices = np.array(overlapping_indices)
    image_paths = np.array(image_paths)
    all_shape_images = []
    all_images = []

    with torch.no_grad():
        for chunk_id in range(len(overlapping_indices)):
            print(f'Processing frames {overlapping_indices[chunk_id][0]} to {overlapping_indices[chunk_id][-1]}')
            image_paths_chunk = image_paths[overlapping_indices[chunk_id]]
            landmarks_chunk = landmarks[overlapping_indices[chunk_id]] if args.crop_face else None

            images_list = []

            for j in range(len(image_paths_chunk)):
                frame = cv2.imread(image_paths_chunk[j])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                kpt = landmarks_chunk[j]

                tform = crop_face(frame, kpt, scale=1.6)
                cropped_image = warp(frame, tform.inverse, output_shape=(224, 224))
                images_list.append(cropped_image.transpose(2, 0, 1))

            images_array = torch.from_numpy(np.array(images_list)).float().to(args.device)

            # Free up memory here to avoid GPU memory overflow
            torch.cuda.empty_cache()
            gc.collect()

            codedict, initial_deca_exp, initial_deca_jaw = spectre.encode(images_array)
            codedict['exp'] += initial_deca_exp
            codedict['pose'][..., 3:] += initial_deca_jaw

            for key in codedict.keys():
                if chunk_id == 0 and chunk_id == len(overlapping_indices) - 1:
                    pass
                elif chunk_id == 0:
                    codedict[key] = codedict[key][:-2]
                elif chunk_id == len(overlapping_indices) - 1:
                    codedict[key] = codedict[key][2:]
                else:
                    codedict[key] = codedict[key][2:-2]

            opdict, visdict = spectre.decode(codedict, rendering=True, vis_lmk=False, return_vis=True)
            all_shape_images.append(visdict['shape_images'].detach().cpu())
            all_images.append(codedict['images'].detach().cpu())

            # Clear CUDA cache and perform garbage collection after each chunk
            torch.cuda.empty_cache()
            gc.collect()

    vid_shape = tensor2video(torch.cat(all_shape_images, dim=0))[2:-2]
    vid_orig = tensor2video(torch.cat(all_images, dim=0))[2:-2]
    grid_vid = np.concatenate((vid_shape, vid_orig), axis=2)

    assert original_video_length == len(vid_shape)

    if args.audio:
        wav, sr = librosa.load(args.input)
        wav = torch.FloatTensor(wav).unsqueeze(0) if len(wav.shape) == 1 else torch.FloatTensor(wav)
        torchvision.io.write_video(f"{videofolder}_shape.mp4", vid_shape, fps=fps, audio_array=wav, audio_codec='aac',audio_fps=sr)
        torchvision.io.write_video(f"{videofolder}_grid.mp4", grid_vid, fps=fps, audio_array=wav, audio_codec='aac', audio_fps=sr)
    else:
        torchvision.io.write_video(f"{videofolder}_shape.mp4", vid_shape, fps=fps)
        torchvision.io.write_video(f"{videofolder}_grid.mp4", grid_vid, fps=fps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    parser.add_argument('-i', '--input', default='examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')
    parser.add_argument('--audio', action='store_true',
                        help='extract audio from the original video and add it to the output video')
    main(parser.parse_args())