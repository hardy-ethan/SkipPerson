import argparse
import numpy as np
import librosa
import soundfile as sf
import torch
from pyannote.audio import Pipeline
import os
import subprocess
from collections import defaultdict
from tqdm import tqdm
import time


def get_audio_tracks(video_path):
    """Detect and return information about audio tracks in the video"""
    print("Detecting audio tracks...")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index:stream_tags=language,title:stream=channels", "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True
    )

    tracks = []
    for i, line in enumerate(result.stdout.strip().split("\n")):
        if not line:
            continue
        parts = line.split(",")
        index = parts[0]
        info = f"Track {i + 1} (stream {index})"

        # Add language and title if available
        if len(parts) > 1 and parts[1]:
            info += f", Language: {parts[1]}"
        if len(parts) > 2 and parts[2]:
            info += f", Title: {parts[2]}"

        # Add channel info
        if len(parts) > 3:
            channels = parts[3]
            info += f", Channels: {channels}"

        tracks.append(info)

    return tracks


def extract_audio_from_video(video_path, stream_index=None):
    """Extract audio from video file using ffmpeg for diarization purposes only"""
    print("Extracting audio from video for diarization...")
    audio_path = os.path.splitext(video_path)[0] + "_TEMP124133audio.wav"

    # Check for multiple audio tracks
    if stream_index is None:
        tracks = get_audio_tracks(video_path)

        if not tracks:
            print("No audio tracks found in the video.")
            return None

        if len(tracks) > 1:
            print("Multiple audio tracks detected:")
            for i, info in enumerate(tracks):
                print(f"{i + 1}. {info}")

            selection = int(input(f"Select audio track to use (1-{len(tracks)}): "))
            stream_index = selection - 1

    # Extract the selected audio track
    cmd = ["ffmpeg", "-y", "-i", video_path, "-q:a", "0"]
    if stream_index is not None:
        cmd.extend(["-map", f"0:a:{stream_index}"])
    else:
        cmd.extend(["-map", "a"])
    cmd.append(audio_path)

    subprocess.run(cmd, check=True)
    return audio_path


def cut_video_by_segments(video_path, segments_to_keep, output_path):
    """Cut video based on segments_to_keep"""
    print("Cutting video based on kept segments...")

    select_string = f"'{'+'.join(map(lambda segment: f'between(t,{segment[0]},{segment[1]})', segments_to_keep))}'"

    # Cut video
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"select={select_string},setpts=N/FRAME_RATE/TB",
        "-af", f"aselect={select_string},asetpts=N/SR/TB",
        output_path
    ], check=True)

    return output_path


def process_media(
    media_path, min_speakers, speakers_to_remove=None, auth_token=None, audio_track=None
):
    """Process media file to remove specific speakers' segments"""
    start_time = time.time()

    # Determine if input is audio or video
    file_ext = os.path.splitext(media_path)[1].lower()
    is_video = file_ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    if not is_video:
        print("This script now only supports video files.")
        return None

    # Extract audio for diarization
    audio_path = extract_audio_from_video(media_path, audio_track)
    if audio_path is None:
        print("Failed to extract audio. Exiting.")
        return None

    # Load audio for duration info
    print("Loading audio file...")
    y, sr = librosa.load(audio_path, sr=None)
    audio_duration = len(y) / sr
    print(f"Audio duration: {audio_duration:.2f} seconds")

    # Perform diarization
    print(
        f"Performing speaker diarization (est. time: {audio_duration/10:.1f}-{audio_duration/5:.1f} minutes)..."
    )
    print("This may take a while for longer files...")

    diarization_start = time.time()
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=auth_token
    ).to(torch.device("cuda"))
    diarization = pipeline(audio_path, min_speakers=min_speakers)
    diarization_time = time.time() - diarization_start
    print(f"Diarization completed in {diarization_time:.1f} seconds")

    # Process diarization results
    print("Processing diarization results...")

    # Identify speakers
    unique_speakers = set()
    for turn, _, speaker in tqdm(
        list(diarization.itertracks(yield_label=True)), desc="Identifying speakers"
    ):
        unique_speakers.add(speaker)

    speakers = sorted(list(unique_speakers))
    print(f"Detected {len(speakers)} speakers")

    # Let user identify which speakers to remove
    if speakers_to_remove is None or len(speakers_to_remove) == 0:
        speaker_segments = defaultdict(list)

        print("Collecting speaker segments...")
        for turn, _, speaker in tqdm(
            list(diarization.itertracks(yield_label=True)), desc="Processing segments"
        ):
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

        # Allow selection of multiple speakers
        speaker_input = input(f"Enter the speaker(s) to remove (1-{len(speakers)}, comma-separated for multiple): ")
        speaker_indices = [int(idx.strip()) - 1 for idx in speaker_input.split(',')]
        target_speakers = [speakers[idx] for idx in speaker_indices]
    else:
        # Convert 1-based indices to 0-based
        speaker_indices = [idx - 1 for idx in speakers_to_remove]
        target_speakers = [speakers[idx] for idx in speaker_indices]

    # Process diarization to get segments to remove
    print("Identifying segments to remove...")
    segments_to_remove = []
    total_segments = 0
    total_remove = 0

    for turn, _, speaker in tqdm(
        list(diarization.itertracks(yield_label=True)), desc="Processing segments"
    ):
        total_segments += 1
        if speaker in target_speakers:
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
    speakers_str = "multiple_speakers" if len(target_speakers) > 1 else "person_B"
    output_path = os.path.splitext(media_path)[0] + f"_no_{speakers_str}" + file_ext
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
    parser.add_argument(
        "min_speakers", type=int, help="Minimum number of speakers in the media"
    )
    parser.add_argument(
        "--speakers", type=int, nargs='+', default=None,
        help="Speaker indices to remove (1-based)"
    )
    parser.add_argument(
        "--auth_token",
        type=str,
        required=True,
        help="Hugging Face token for pyannote.audio",
    )
    parser.add_argument(
        "--audio_track",
        type=int,
        default=None,
        help="Audio track to use (0-based stream index)",
    )

    args = parser.parse_args()
    output_path = process_media(
        args.media_path,
        args.min_speakers,
        args.speakers,
        args.auth_token,
        args.audio_track,
    )
    print(f"Processed file saved to: {output_path}")


if __name__ == "__main__":
    main()
