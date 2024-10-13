from helpers import *
import sys

def main(video_url):
    video_id = extract_video_id(video_url)
    if video_id:
        print(f"Extracted video ID: {video_id}")
        download_video(video_id)
        download_subtitles(video_id)
    else:
        print("Invalid YouTube URL. Could not extract video ID.")

    lang = select_language()
    gender = select_gender()

    # Load the SRT file
    subs = load_first_subtitle()

    updated_subs_array1, combined_text1, all_words1 = extract_subs(subs)

    punctuated_text1 = punctuate_text(combined_text1)

    print(punctuated_text1)
    sentences_with_indices1 = get_sentences_with_indices(punctuated_text1, all_words1)

    sentences_with_timestamps1 = get_sentences_with_timestamps(sentences_with_indices1, updated_subs_array1)

    # Calculate total duration
    total_duration = calculate_duration(sentences_with_timestamps1)

    sentences_with_indices1 = get_sentences_with_indices(punctuated_text1, all_words1)

    sentences_with_timestamps1 = get_sentences_with_timestamps(sentences_with_indices1, updated_subs_array1)

    base_audio = combined_audio(sentences_with_timestamps1, target_language=lang['lang_target'], language_code=lang['lang_code'], speaking_rate=1.3, gender=gender['gender'])
    # After creating and processing base_audio
    output_filename = "output_audio.mp3"
    base_audio.export(output_filename, format="mp3")
    try:
        # Play the audio
        play_audio_video("output_audio.mp3", 'video.mkv')
    except FileNotFoundError:
        print("Error: The file 'output_audio.mp3' was not found.")
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")

def select_language():
    all_languages = language_code_from_language()

    for index, lang in enumerate(all_languages):
        print(f"{index}: {lang['human_readable_language']}")

    while True:
        try:
            selection = int(input("\nEnter the number of your desired language: "))
            if 0 <= selection < len(all_languages):
                return all_languages[selection]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def select_gender():
    all_genders = get_all_gender()

    for index, gender in enumerate(all_genders):
        print(f"{index}: {gender['name']}")

    while True:
        try:
            selection = int(input("\nEnter the number of your desired gender: "))
            if 0 <= selection < len(all_genders):
                return all_genders[selection]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
        main(video_url)
    else:
        print("Please provide a language as a command-line argument.")
        # Optionally, you can call select_language() here if no argument is provided
        # language = select_language()
        # main(language)

    # select_language()
    # play_audio_video("output_audio.mp3", 'video.mkv')
