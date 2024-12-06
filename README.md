
# **SPECTRE: Visual Speech-Aware Perceptual 3D Facial Expression Reconstruction from Videos**

Official PyTorch implementation of the paper:  
**Visual Speech-Aware Perceptual 3D Facial Expression Reconstruction from Videos**  
Authors: Panagiotis P. Filntisis, George Retsinas, Foivos Paraperas-Papantoniou, Athanasios Katsamanis, Anastasios Roussos, and Petros Maragos  
Published on: arXiv 2022  

---

## **Overview**

Our method performs visual-speech-aware 3D reconstruction so that speech perception from the original footage is preserved in the reconstructed talking head. For each example, the word/phrase being said is included.

---

## **Installation**

### Requirements: 
- Python = 3.10.11
- Cuda = 12.4 
`` Download the Cuda version from: https://developer.nvidia.com/cuda-12-4-0-download-archive ``


### Demo Using Kaggle 

```
https://www.kaggle.com/code/niwant/ccn-spectre-project
```
Use the following kaggle linke to run the Model for Demo on Kaggle . 


### Installation from Spectre Local setup Notebook
 All the following installation can be directly performed by the `spectre_local_setup.ipynb` jupyter file.
 - Open the Jupyter File in vscode 
 - click on Run all cells to run the intsallation 

`` Note : If there is any failure during the installation through the notebook run the command from the failed cell in the terminal  ``

### Clone the Repository
```bash
git clone --recurse-submodules -j4 https://github.com/ibrahim-anas/SPECTRE.git
cd spectre
```

### Install Dependencies
Ensure you have Python 3.6 or higher. Install the required libraries with the following commands:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install fvcore iopath scikit_image scipy kornia chumpy librosa av loguru tensorboard pytorch_lightning opencv-python phonemizer jiwer gdown yacs numpy==1.23.5 gradio
```

### Install External Packages

1. Install **face_detection**:
   ```bash
   cd external/face_detection
   git lfs pull
   pip install -e .
   ```

2. Install **face_alignment**:
   ```bash
   cd external/face_alignment
   git lfs pull
   pip install -e .
   ```

---

## **Download Models**

Use the following command to download the FLAME model and the pretrained SPECTRE model:
```bash
pip install gdown
bash quick_install.sh
```

---

## **Running the Demo**

### Demo via Command Line
To test the demo on the sample video provided:
```bash
python demo.py --input samples/MEAD/M003_level_1_disgusted_015.mp4 --audio
```

### Demo via Gradio Interface
You can also interact with the system using a Gradio interface. The interface allows you to upload a video, process it with SPECTRE, and download the reconstructed output.

1. **Run the Gradio Demo**:
    ```python
    import gradio as gr
    import subprocess
    import os

    def process_video(input_video):
        input_video_path = input_video

        directory, file_name = os.path.split(input_video_path)
        name, ext = os.path.splitext(file_name)

        # Modify the file name
        new_file_name = f"{name}_grid{ext}"
        output_video_path = os.path.join(directory, new_file_name)

        try:
            # Execute command
            command = ["python", "demo.py", "--input", input_video_path, "--audio"]
            subprocess.run(command, check=True)
            
            # Check if the output file was created
            if os.path.exists(output_video_path):
                return output_video_path  # Return the path to the output video
            else:
                return "Error: Output video could not be generated."
        except subprocess.CalledProcessError as e:
            return f"Error: {str(e)}"

    demo = gr.Interface(
        fn=process_video,
        inputs=gr.Video(interactive=True),
        outputs=gr.Video(format="mp4"),  # Ensure output is treated as a video file path
        title="Spectre Demo",
        description="Upload a video to Spectre, and view the output."
    )

    demo.launch(share=True)
    ```

2. **Upload a Video**  
   Interact with the Gradio interface in your browser. Upload a video, and it will process the input and return a reconstructed output.

---

## **Notes**
- **FFmpeg** is required for video processing. Ensure it is installed in your environment.
- If using a GPU, ensure that CUDA is correctly configured for PyTorch.

--- 

Feel free to reach out for further assistance!
