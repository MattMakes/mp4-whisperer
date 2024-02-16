import subprocess
import whisper

def extract_audio(video_path, audio_path):
    """
    Extracts the audio from the video file using FFmpeg.
    
    Args:
    video_path (str): Path to the input video file.
    audio_path (str): Path where the extracted audio file will be saved.
    """
    command = f"ffmpeg -i {video_path} -ac 1 -ar 16000 {audio_path}"
    subprocess.run(command, shell=True, check=True)

def transcribe_audio(audio_path):
    """
    Transcribes the audio file using Whisper AI.
    
    Args:
    audio_path (str): Path to the audio file to transcribe.
    
    Returns:
    str: Transcribed text.
    """
    # Load the model
    model = whisper.load_model("large")
    
    # Transcribe the audio
    result = model.transcribe(audio_path)
    return result["text"]

def write_transcription_to_file(transcription, output_file):
    """
    Writes the transcription to a text file.
    
    Args:
    transcription (str): The transcribed text.
    output_file (str): Path to the output text file.
    """
    with open(output_file, "w") as file:
        file.write(transcription)

if __name__ == "__main__":
    file_name = "ProblemAnalysis_FitnessFunctions_Pt2"
    video_path = "./video/" + file_name + ".mp4"
    audio_path = "./temp/" + file_name + ".wav"
    output_file = "./transcriptions/" + file_name + ".txt"
    
    # Extract audio from the video
    extract_audio(video_path, audio_path)
    
    # Transcribe the audio
    transcription = transcribe_audio(audio_path)
    
    # Write the transcription to a file
    write_transcription_to_file(transcription, output_file)
    
    print("Transcription completed and written to", output_file)
