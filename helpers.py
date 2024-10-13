import subprocess
import pysrt
import re
import pysrt
import nltk
from nltk.tokenize import word_tokenize
import os
from dotenv import load_dotenv
import openai
import difflib
import os
import glob
import subprocess
import html
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

# Load environment variables from .env file
load_dotenv()


# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# openai_api_base = "https://run-execution-izdjkfexc0n4-run-execution-8000.sanjose.oracle-cluster.vessl.ai/"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

# Download NLTK data if not already present
nltk.download('punkt', quiet=True)

import re
from urllib.parse import urlparse, parse_qs

def extract_video_id(video_url):
    # Method 1: Using regular expression
    youtube_regex = r'(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})'
    match = re.search(youtube_regex, video_url)
    if match:
        return match.group(1)

    # Method 2: Using urllib.parse
    parsed_url = urlparse(video_url)
    if parsed_url.hostname in ('youtu.be', 'youtube.com', 'www.youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
        elif parsed_url.path.startswith(('/embed/', '/v/')):
            return parsed_url.path.split('/')[2]

    # If no match found
    return None

def download_video(video_id):
    command = [
        "yt-dlp",
        "--output", "video.%(ext)s",
        "--merge-output-format", "mkv",
        "--force-overwrites",
        f'https://www.youtube.com/watch?v={video_id}'
    ]

    try:
        result = subprocess.run(command, check=True)
        print("Video downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print("Error output:")

def download_subtitles(video_id):
    for file in glob.glob("transcript.*.srt"):
        os.remove(file)
        print(f"Deleted existing file: {file}")

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
        print("Subtitles downloaded and converted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print("Error output:")

def load_first_subtitle():
    # Find all files matching the pattern
    subtitle_files = glob.glob('transcript.*.srt')

    if not subtitle_files:
        print("No subtitle files found.")
        return None

    # Sort the files to ensure consistent behavior
    subtitle_files.sort()

    # Get the first file
    first_subtitle_file = subtitle_files[0]

    print(f"Loading subtitle file: {first_subtitle_file}")

    # Load and return the subtitles
    return pysrt.open(first_subtitle_file)


import pysrt
import re

def extract_subs(subs):
    # Remove <font> tags from each subtitle's text
    for sub in subs:
        sub.text = re.sub(r'<font[^>]*>|</font>', '', sub.text)

    subs_array = [(sub.start, sub.end, sub.text) for sub in subs]

    # Combine all subtitle text into a single string
    combined_text = ' '.join(sub[2] for sub in subs_array)

    # Tokenize the combined text
    all_words = word_tokenize(combined_text.lower())

    # Create a new array to store the updated subtitle information
    updated_subs_array = []

    # Keep track of the current word index
    current_word_index = 0

    for start, end, text in subs_array:
        # Tokenize the current subtitle text
        sub_words = word_tokenize(text.lower())

        # Find the indices of the words in this subtitle
        word_indices = []
        for word in sub_words:
            while current_word_index < len(all_words) and all_words[current_word_index] != word:
                current_word_index += 1
            if current_word_index < len(all_words):
                word_indices.append(current_word_index)
                current_word_index += 1

        # Add the updated tuple to the new array
        updated_subs_array.append((start, end, text, word_indices))

    return updated_subs_array, combined_text, all_words


def punctuate_text(text):
    llm = OpenAI(model="gpt-3.5-turbo")
    messages = [
        ChatMessage(
            role="system", content="You are a helpful assistant that accurately punctuates text."
        ),
        ChatMessage(role="user", content=f"Please punctuate the following text accurately:\n\n{text}"),
    ]


    resp = llm.chat(
        messages=messages,
        temperature=0
    )

    return resp.message.content

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import difflib

def get_sentences_with_indices(punctuated_text, all_words):
    # Split the punctuated text into sentences
    sentences = sent_tokenize(punctuated_text)

    # Tokenize the entire punctuated text
    punctuated_words = word_tokenize(punctuated_text)

    # Create a new list to store sentences with their word indices
    sentences_with_indices = []

    # Loop through sentences and find their indices in punctuated_words
    for sentence in sentences:
        sent_tokens = word_tokenize(sentence)
        sentence_indices = []

        # Find the start and end indices of the sentence in punctuated_words
        for i in range(len(punctuated_words) - len(sent_tokens) + 1):
            if punctuated_words[i:i+len(sent_tokens)] == sent_tokens:
                sentence_indices = list(range(i, i + len(sent_tokens)))
                break

        sentences_with_indices.append([sentence, sentence_indices])

    # Print the first few elements of sentences_with_indices to verify
    for i, (sentence, indices) in enumerate(sentences_with_indices[:3]):
        print(f"Sentence {i + 1}:")
        print(f"Text: {sentence}")
        print(f"Indices: {indices}")
        print()


    # Assuming all_words and punctuated_words are already defined

    # Create a SequenceMatcher object
    matcher = difflib.SequenceMatcher(None, all_words, punctuated_words)

    # Get the opcodes which describe the differences between the sequences
    opcodes = matcher.get_opcodes()

    # Print the differences
    print("Differences between all_words and punctuated_words:")
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            print(f"Equal:      {' '.join(all_words[i1:i2])}")
        elif tag == 'delete':
            print(f"Delete from all_words:      {' '.join(all_words[i1:i2])}")
        elif tag == 'insert':
            print(f"Insert into punctuated_words: {' '.join(punctuated_words[j1:j2])}")
        elif tag == 'replace':
            print(f"Replace: {' '.join(all_words[i1:i2])} -> {' '.join(punctuated_words[j1:j2])}")

    # Print some statistics
    print("\nStatistics:")
    print(f"Ratio of similarity: {matcher.ratio():.2%}")
    print(f"Number of all_words: {len(all_words)}")
    print(f"Number of punctuated_words: {len(punctuated_words)}")
    # Now sentences_with_indices has the structure you requested


    # Assuming punctuated_text, all_words, and punctuated_words are already defined

    # Create a SequenceMatcher object
    matcher = difflib.SequenceMatcher(None, all_words, punctuated_words)

    # Get the opcodes which describe the differences between the sequences
    opcodes = matcher.get_opcodes()

    # Create a mapping from punctuated_words indices to all_words indices
    punctuated_to_original = {}
    original_index = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag in ['equal', 'replace']:
            for punctuated_index in range(j1, j2):
                punctuated_to_original[punctuated_index] = original_index
                original_index += 1
        elif tag == 'delete':
            original_index += (i2 - i1)
        # We don't need to do anything for 'insert' as it doesn't affect all_words indices

    # Split the punctuated text into sentences
    sentences = sent_tokenize(punctuated_text)

    # Create a new list to store sentences with their word indices
    sentences_with_indices = []

    # Loop through sentences and find their indices in punctuated_words and all_words
    for sentence in sentences:
        sent_tokens = word_tokenize(sentence)
        punctuated_indices = []
        original_indices = []

        # Find the start and end indices of the sentence in punctuated_words
        for i in range(len(punctuated_words) - len(sent_tokens) + 1):
            if punctuated_words[i:i+len(sent_tokens)] == sent_tokens:
                punctuated_indices = list(range(i, i + len(sent_tokens)))
                original_indices = [punctuated_to_original.get(idx, None) for idx in punctuated_indices]
                break

        sentences_with_indices.append([sentence, punctuated_indices, original_indices])

    # Print the first few elements of sentences_with_indices to verify
    for i, (sentence, punctuated_indices, original_indices) in enumerate(sentences_with_indices):
        print(f"Sentence {i + 1}:")
        print(f"Text: {sentence}")
        print(f"Punctuated indices: {punctuated_indices}")
        print(f"Original indices: {original_indices}")
        print()

    return sentences_with_indices

# Assuming sentences_with_indices and updated_subs_array are already defined

def get_sentences_with_timestamps(sentences_with_indices, updated_subs_array):
    # Create a new list to store the updated sentences with indices
    sentences_with_all_indices = []

    for sentence, punctuated_indices, original_indices in sentences_with_indices:
        matching_sub_indices = []

        # Find matching subtitles
        for sub_index, (_, _, _, sub_word_indices) in enumerate(updated_subs_array):
            # Check if there's any overlap between the sentence words and subtitle words
            if any(index in sub_word_indices for index in original_indices if index is not None):
                matching_sub_indices.append(sub_index)

        # Add the matching subtitle indices as the fourth element
        sentences_with_all_indices.append([sentence, punctuated_indices, original_indices, matching_sub_indices])

    sentences_with_timestamps = []
    # Print the first few elements of sentences_with_all_indices to verify
    for i, (sentence, punctuated_indices, original_indices, matching_sub_indices) in enumerate(sentences_with_all_indices):
        if len(matching_sub_indices) > 0:
            sentences_with_timestamps.append((updated_subs_array[matching_sub_indices[0]][0], updated_subs_array[matching_sub_indices[-1]][1], sentence))
            print(f"Sentence {i + 1}:")
            print(f"Text: {sentence}")
            print(f"Punctuated indices: {punctuated_indices}")
            print(f"Original indices: {original_indices}")
            print(f"Matching subtitle indices: {matching_sub_indices}")

            print(updated_subs_array[matching_sub_indices[0]][0], updated_subs_array[matching_sub_indices[-1]][1])

    return sentences_with_timestamps

def calculate_duration(sentences_with_timestamps):
  # Get the start time of the first sentence and end time of the last sentence
  start_time = sentences_with_timestamps[0][0]
  end_time = sentences_with_timestamps[-1][1]

    # Convert times to seconds
  start_seconds = start_time.hours * 3600 + start_time.minutes * 60 + start_time.seconds + start_time.milliseconds / 1000
  end_seconds = end_time.hours * 3600 + end_time.minutes * 60 + end_time.seconds + end_time.milliseconds / 1000

  # Calculate duration
  return end_seconds - start_seconds



import os
import io
from google.cloud import translate_v2 as translate
from google.cloud import texttospeech
import pygame

# Set the path to your Google Cloud credentials JSON file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/kenneth/Projects/tsd-enterprise/gcp-key/key.json'

def translate_text(text, target_language='fr'):
    # Remove text within square brackets
    text_without_brackets = re.sub(r'\[.*?\]', '', text)

    translate_client = translate.Client()
    result = translate_client.translate(text_without_brackets, target_language=target_language)

    # Unescape HTML entities
    translated_text = html.unescape(result['translatedText'])

    return translated_text

def get_all_gender():
    return [
      {'gender': texttospeech.SsmlVoiceGender.NEUTRAL, 'name': 'Neutral'},
      {'gender': texttospeech.SsmlVoiceGender.FEMALE, 'name': 'Female'},
      {'gender': texttospeech.SsmlVoiceGender.MALE, 'name': 'Male'},
    ]

def text_to_speech(text, language_code='fr-FR', speaking_rate=1.0, gender=texttospeech.SsmlVoiceGender.NEUTRAL):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=gender
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response.audio_content

def language_code_from_language():
  langs = [{'human_readable_language': 'Afrikaans',
        'lang_code': 'af-ZA',
        'lang_target': 'af'},
      {'human_readable_language': 'Amharic',
        'lang_code': 'am-ET',
        'lang_target': 'am'},
      {'human_readable_language': 'Arabic',
        'lang_code': 'ar-XA',
        'lang_target': 'ar'},
      {'human_readable_language': 'Basque',
        'lang_code': 'eu-ES',
        'lang_target': 'eu'},
      {'human_readable_language': 'Bengali',
        'lang_code': 'bn-IN',
        'lang_target': 'bn'},
      {'human_readable_language': 'Bulgarian',
        'lang_code': 'bg-BG',
        'lang_target': 'bg'},
      {'human_readable_language': 'Cantonese',
        'lang_code': 'yue-HK',
        'lang_target': 'yue'},
      {'human_readable_language': 'Catalan',
        'lang_code': 'ca-ES',
        'lang_target': 'ca'},
      {'human_readable_language': 'Chinese (Simplified)',
         'lang_code': 'zh-CN',
         'lang_target': 'zh-CN'},
      {'human_readable_language': 'Chinese (Traditional)',
         'lang_code': 'zh-TW',
         'lang_target': 'zh-TW'},
      {'human_readable_language': 'Czech',
        'lang_code': 'cs-CZ',
        'lang_target': 'cs'},
      {'human_readable_language': 'Danish',
        'lang_code': 'da-DK',
        'lang_target': 'da'},
      {'human_readable_language': 'Dutch',
        'lang_code': 'nl-BE',
        'lang_target': 'nl'},
      {'human_readable_language': 'English',
        'lang_code': 'en-AU',
        'lang_target': 'en'},
      {'human_readable_language': 'Finnish',
        'lang_code': 'fi-FI',
        'lang_target': 'fi'},
      {'human_readable_language': 'French',
        'lang_code': 'fr-CA',
        'lang_target': 'fr'},
      {'human_readable_language': 'Galician',
        'lang_code': 'gl-ES',
        'lang_target': 'gl'},
      {'human_readable_language': 'German',
        'lang_code': 'de-DE',
        'lang_target': 'de'},
      {'human_readable_language': 'Greek',
        'lang_code': 'el-GR',
        'lang_target': 'el'},
      {'human_readable_language': 'Gujarati',
        'lang_code': 'gu-IN',
        'lang_target': 'gu'},
      {'human_readable_language': 'Hebrew',
        'lang_code': 'he-IL',
        'lang_target': 'he'},
      {'human_readable_language': 'Hindi',
        'lang_code': 'hi-IN',
        'lang_target': 'hi'},
      {'human_readable_language': 'Hungarian',
        'lang_code': 'hu-HU',
        'lang_target': 'hu'},
      {'human_readable_language': 'Icelandic',
        'lang_code': 'is-IS',
        'lang_target': 'is'},
      {'human_readable_language': 'Indonesian',
        'lang_code': 'id-ID',
        'lang_target': 'id'},
      {'human_readable_language': 'Italian',
        'lang_code': 'it-IT',
        'lang_target': 'it'},
      {'human_readable_language': 'Japanese',
        'lang_code': 'ja-JP',
        'lang_target': 'ja'},
      {'human_readable_language': 'Kannada',
        'lang_code': 'kn-IN',
        'lang_target': 'kn'},
      {'human_readable_language': 'Korean',
        'lang_code': 'ko-KR',
        'lang_target': 'ko'},
      {'human_readable_language': 'Latvian',
        'lang_code': 'lv-LV',
        'lang_target': 'lv'},
      {'human_readable_language': 'Lithuanian',
        'lang_code': 'lt-LT',
        'lang_target': 'lt'},
      {'human_readable_language': 'Malay',
        'lang_code': 'ms-MY',
        'lang_target': 'ms'},
      {'human_readable_language': 'Malayalam',
        'lang_code': 'ml-IN',
        'lang_target': 'ml'},
      {'human_readable_language': 'Marathi',
        'lang_code': 'mr-IN',
        'lang_target': 'mr'},
      {'human_readable_language': 'Polish',
        'lang_code': 'pl-PL',
        'lang_target': 'pl'},
      {'human_readable_language': 'Portuguese',
        'lang_code': 'pt-BR',
        'lang_target': 'pt'},
      {'human_readable_language': 'Punjabi',
        'lang_code': 'pa-IN',
        'lang_target': 'pa'},
      {'human_readable_language': 'Romanian',
        'lang_code': 'ro-RO',
        'lang_target': 'ro'},
      {'human_readable_language': 'Russian',
        'lang_code': 'ru-RU',
        'lang_target': 'ru'},
      {'human_readable_language': 'Serbian',
        'lang_code': 'sr-RS',
        'lang_target': 'sr'},
      {'human_readable_language': 'Slovak',
        'lang_code': 'sk-SK',
        'lang_target': 'sk'},
      {'human_readable_language': 'Spanish',
        'lang_code': 'es-ES',
        'lang_target': 'es'},
      {'human_readable_language': 'Swedish',
        'lang_code': 'sv-SE',
        'lang_target': 'sv'},
      {'human_readable_language': 'Tamil',
        'lang_code': 'ta-IN',
        'lang_target': 'ta'},
      {'human_readable_language': 'Telugu',
        'lang_code': 'te-IN',
        'lang_target': 'te'},
      {'human_readable_language': 'Thai', 'lang_code': 'th-TH', 'lang_target': 'th'},
      {'human_readable_language': 'Turkish',
        'lang_code': 'tr-TR',
        'lang_target': 'tr'},
      {'human_readable_language': 'Ukrainian',
        'lang_code': 'uk-UA',
        'lang_target': 'uk'},
      {'human_readable_language': 'Urdu', 'lang_code': 'ur-IN', 'lang_target': 'ur'},
      {'human_readable_language': 'Vietnamese',
        'lang_code': 'vi-VN',
        'lang_target': 'vi'}
        ]
  return langs

import os
import io
from google.cloud import translate_v2 as translate
from google.cloud import texttospeech
import pygame
from pydub import AudioSegment

def combined_audio(sentences_with_timestamps, target_language='fr', language_code='fr-FR', speaking_rate=1.0, gender=texttospeech.SsmlVoiceGender.NEUTRAL):
  total_duration = calculate_duration(sentences_with_timestamps)

  # Calculate total duration (assuming this has been done as in the previous example)
  total_duration_ms = int(total_duration * 1000)  # Convert to milliseconds

  print(total_duration_ms)

  # Create a silent base audio
  base_audio = AudioSegment.silent(duration=total_duration_ms)

  # Loop through sentences_with_timestamps1
  for start_time, end_time, sentence in sentences_with_timestamps:
      # Translate the sentence to French
      french_translation = translate_text(sentence, target_language=target_language)

      print(french_translation)
      # Convert French translation to speech
      audio_content = text_to_speech(french_translation, language_code=language_code, speaking_rate=speaking_rate, gender=gender)

      # Create an AudioSegment from the speech audio content
      sentence_audio = AudioSegment.from_mp3(io.BytesIO(audio_content))

      # Calculate the start position in milliseconds
      start_ms = (start_time.hours * 3600 + start_time.minutes * 60 + start_time.seconds) * 1000 + start_time.milliseconds

      # Overlay the sentence audio onto the base audio at the correct position
      base_audio = base_audio.overlay(sentence_audio, position=start_ms)

  return base_audio

import io
import pygame
import io
import sys
import cv2
import time

def play_audio_video(audio_file_path, video_file_path):

    cap = cv2.VideoCapture(video_file_path)

    # Get the framerate
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video framerate: {fps}")

    # Calculate delay for each frame
    frame_delay = 1 / fps

    start_time = time.time()
    prev_time = start_time

    i = 0
    target_accumulated_delay = 0


    with open(audio_file_path, "rb") as audio_file:
        audio_content = audio_file.read()

    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the audio from the response
    audio_file = io.BytesIO(audio_content)
    pygame.mixer.music.load(audio_file)

    # Play the audio
    pygame.mixer.music.play()


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Video', frame)

        # if pygame.mixer.music.get_busy():
        #     pygame.time.Clock().tick(1)

        target_accumulated_delay += frame_delay

        actual_accumulated_delay = time.time() - start_time

        print(target_accumulated_delay, actual_accumulated_delay, i)
        i += 1
        if target_accumulated_delay - actual_accumulated_delay > 0:
            wait_time = target_accumulated_delay - actual_accumulated_delay
            time.sleep(wait_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    total_playback_time = end_time - start_time

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()
    pygame.mixer.quit()

    print(f"Total playback time: {total_playback_time:.2f} seconds")
