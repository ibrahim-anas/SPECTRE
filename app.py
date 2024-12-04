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
    print(output_video_path)

    try:
        # Execute command
        command = ["python", "demo.py", "--input", input_video_path]
        subprocess.run(command, check=True)
        
        # Check if the output file was created
        if os.path.exists(output_video_path):
            return output_video_path # Return the path to the output video
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
