import subprocess
import pysrt

def download_subtitles(video_id):
    command = [
        "yt-dlp",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-format", "ttml",
        "--convert-subs", "srt",
        "--output", "transcript.%(ext)s",
        video_id
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Command output:")
        print(result.stdout)
        print("Subtitles downloaded and converted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print("Error output:")
        print(e.stderr)

# Function to process subtitles
def process_subtitles(srt_file):
    subs = pysrt.open(srt_file)
    # Add your subtitle processing logic here
    print(f"Loaded {len(subs)} subtitles from {srt_file}")
    # Example: Print the first subtitle
    if subs:
        print("First subtitle:", subs[0].text)

if __name__ == "__main__":
    video_id = '8MzQjtIkF5g'
    download_subtitles(video_id)

    # Process the downloaded subtitles
    srt_file = 'transcript.en.srt'
    process_subtitles(srt_file)
