import argparse
import numpy as np
import librosa
import soundfile as sf
from pyannote.audio import Pipeline
import os
import subprocess
from collections import defaultdict
from tqdm import tqdm
import time

def extract_audio_from_video(video_path):
    """Extract audio from video file using ffmpeg for diarization purposes only"""
    print("Extracting audio from video for diarization...")
    audio_path = os.path.splitext(video_path)[0] + "_TEMP124133audio.wav"
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path], 
                   check=True)
    return audio_path

def cut_video_by_segments(video_path, segments_to_keep, output_path):
    """Cut video based on segments_to_keep"""
    print("Cutting video based on kept segments...")
    temp_dir = "temp_video_segments"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create temporary video segments
    segment_files = []
    for i, (start_sec, end_sec) in enumerate(tqdm(segments_to_keep, desc="Cutting video segments")):
        duration = end_sec - start_sec
        segment_file = f"{temp_dir}/segment_{i}.mp4"
        segment_files.append(segment_file)
        
        # Cut video segment using ffmpeg
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, 
            "-ss", str(start_sec), "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-c:a", "aac",
            segment_file
        ], check=True)
    
    # Create ffmpeg concat file
    concat_file = f"{temp_dir}/concat_list.txt"
    with open(concat_file, "w") as f:
        for segment_file in segment_files:
            f.write(f"file '{os.path.abspath(segment_file)}'\n")
    
    # Concat video segments
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", concat_file, "-c", "copy", output_path
    ], check=True)
    
    # Clean up temp files
    for file in segment_files + [concat_file]:
        if os.path.exists(file):
            os.remove(file)
    os.rmdir(temp_dir)
    
    return output_path

def process_media(media_path, min_speakers, speaker_to_remove=None, auth_token=None):
    """Process media file to remove a specific speaker's segments"""
    start_time = time.time()
    
    # Determine if input is audio or video
    file_ext = os.path.splitext(media_path)[1].lower()
    is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    if not is_video:
        print("This script now only supports video files.")
        return None
    
    # Extract audio for diarization
    audio_path = extract_audio_from_video(media_path)
    
    # Load audio for duration info
    print("Loading audio file...")
    y, sr = librosa.load(audio_path, sr=None)
    audio_duration = len(y) / sr
    print(f"Audio duration: {audio_duration:.2f} seconds")
    
    # Perform diarization
    print(f"Performing speaker diarization (est. time: {audio_duration/10:.1f}-{audio_duration/5:.1f} minutes)...")
    print("This may take a while for longer files...")
    
    diarization_start = time.time()
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=auth_token)
    diarization = pipeline(audio_path, min_speakers=min_speakers)
    diarization_time = time.time() - diarization_start
    print(f"Diarization completed in {diarization_time:.1f} seconds")
    
    # Process diarization results
    print("Processing diarization results...")
    
    # Identify speakers
    unique_speakers = set()
    for turn, _, speaker in tqdm(list(diarization.itertracks(yield_label=True)), desc="Identifying speakers"):
        unique_speakers.add(speaker)
    
    speakers = sorted(list(unique_speakers))
    print(f"Detected {len(speakers)} speakers")
    
    # Let user identify which speaker to remove
    if speaker_to_remove is None:
        speaker_segments = defaultdict(list)
        
        print("Collecting speaker segments...")
        for turn, _, speaker in tqdm(list(diarization.itertracks(yield_label=True)), desc="Processing segments"):
            speaker_segments[speaker].append((turn.start, turn.end))
        
        # Save sample for each speaker
        print("Creating speaker samples...")
        for i, speaker in enumerate(tqdm(speakers, desc="Saving speaker samples")):
            segment = speaker_segments[speaker][0]
            start_sec, end_sec = segment
            end_sec = min(start_sec + 5, end_sec)
            
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            speaker_audio = y[start_sample:end_sample]
            
            temp_file = f"speaker_{i+1}_sample.wav"
            sf.write(temp_file, speaker_audio, sr)
            print(f"Sample for Speaker {i+1} saved to: {temp_file}")
        
        speaker_idx = int(input(f"Enter the speaker to remove (1-{len(speakers)}): ")) - 1
        target_speaker = speakers[speaker_idx]
    else:
        speaker_idx = speaker_to_remove - 1
        target_speaker = speakers[speaker_idx]
    
    # Process diarization to get segments to remove
    print("Identifying segments to remove...")
    segments_to_remove = []
    total_segments = 0
    total_remove = 0
    
    for turn, _, speaker in tqdm(list(diarization.itertracks(yield_label=True)), desc="Processing segments"):
        total_segments += 1
        if speaker == target_speaker:
            segments_to_remove.append((turn.start, turn.end))
            total_remove += 1
    
    segments_to_remove.sort(key=lambda x: x[0])
    remove_percent = (total_remove / total_segments) * 100
    print(f"Removing {total_remove}/{total_segments} segments ({remove_percent:.1f}%)")
    
    # Convert segments to remove into segments to keep by finding the gaps
    print("Creating segments to keep...")
    segments_to_keep = []
    current_time = 0
    
    for start_sec, end_sec in segments_to_remove:
        # Add segment before the current removal segment (if any)
        if start_sec > current_time:
            segments_to_keep.append((current_time, start_sec))
        current_time = end_sec
    
    # Add final segment after the last removal (if needed)
    if current_time < audio_duration:
        segments_to_keep.append((current_time, audio_duration))
    
    print(f"Keeping {len(segments_to_keep)} segments of video")
    
    # Cut and concat video
    output_path = os.path.splitext(media_path)[0] + "_no_person_B" + file_ext
    cut_video_by_segments(media_path, segments_to_keep, output_path)
    
    # Clean up temp files
    print("Cleaning up temporary files...")
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    for i in range(len(speakers)):
        temp_file = f"speaker_{i+1}_sample.wav"
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    total_time = time.time() - start_time
    print(f"\nProcessing completed in {total_time:.1f} seconds")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Remove a specific speaker from video")
    parser.add_argument("media_path", type=str, help="Path to the video file")
    parser.add_argument("min_speakers", type=int, help="Minimum number of speakers in the media")
    parser.add_argument("--speaker", type=int, default=None, help="Speaker index to remove (1-based)")
    parser.add_argument("--auth_token", type=str, required=True, help="Hugging Face token for pyannote.audio")
    
    args = parser.parse_args()
    output_path = process_media(args.media_path, args.min_speakers, args.speaker, args.auth_token)
    print(f"Processed file saved to: {output_path}")

if __name__ == "__main__":
    main()
