from json import dumps
from TTS.api import TTS
import os
import torch
import time
import re
import tempfile
import uuid
from datetime import datetime
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.configs.xtts_config import XttsAudioConfig
from TTS.tts.configs.xtts_config import XttsArgs
from TTS.tts.configs.shared_configs import BaseAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from pydub import AudioSegment
# Add necessary global for safe loading
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs, BaseAudioConfig])

# Default model
DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

def get_device():
    """
    Return the appropriate device (CUDA if available, otherwise CPU).
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def list_available_speakers(model_name=DEFAULT_MODEL):
    """
    List all available speakers for the specified TTS model.
    
    :param model_name: Name of the TTS model to use
    :return: List of available speakers
    """
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize TTS and move to device
    tts = TTS(model_name=model_name).to(device)
    
    # Check if model has predefined speaker references
    speakers = []
    if hasattr(tts.synthesizer.tts_model, 'speaker_manager') and hasattr(tts.synthesizer.tts_model.speaker_manager, 'speakers'):
        speakers = tts.synthesizer.tts_model.speaker_manager.speakers
    
    print("Available speakers:")
    for speaker in speakers:
        print(f"- {speaker}")
    
    return speakers

def parse_mixed_language_text(text):
    """
    Parse mixed language text with language tags and pause markers.
    
    Language tags like <de> or <en> set the language until another language tag is found.
    Pause markers use format {PAUSE=400} where 400 is milliseconds.
    The notations themselves are removed from the final text segments.
    
    :param text: Text containing language tags and pause markers
    :return: List of tuples (text segment, language, pause_duration)
    """
    segments = []
    current_text = ""
    current_lang = "de"  # Default language
    
    # Pattern for language tags and pauses
    lang_pattern = re.compile(r'<([a-z]{2})>')
    pause_pattern = re.compile(r'\{PAUSE=(\d+)\}')
    
    # Process text character by character to handle tags properly
    i = 0
    while i < len(text):
        # Check for language tag
        lang_match = lang_pattern.match(text[i:])
        if lang_match:
            # If there's accumulated text, add it to segments
            if current_text.strip():
                segments.append((current_text.strip(), current_lang, 0))
                current_text = ""
            
            # Update current language
            current_lang = lang_match.group(1)
            i += len(lang_match.group(0))
            continue
        
        # Check for pause marker
        pause_match = pause_pattern.match(text[i:])
        if pause_match:
            # If there's accumulated text, add it to segments
            if current_text.strip():
                segments.append((current_text.strip(), current_lang, 0))
                current_text = ""
            
            # Add pause segment
            pause_duration = int(pause_match.group(1))
            segments.append(("", "", pause_duration))
            i += len(pause_match.group(0))
            continue
        
        # Regular character
        current_text += text[i]
        i += 1
    
    # Add remaining text if any
    if current_text.strip():
        segments.append((current_text.strip(), current_lang, 0))
    
    return segments

def create_silence(duration_ms, output_file):
    """
    Create a silence audio file of specified duration.
    
    :param duration_ms: Duration in milliseconds
    :param output_file: Path to save the silence audio file
    """
    silence = AudioSegment.silent(duration=duration_ms)
    silence.export(output_file, format="wav")

def merge_audio_files(audio_files, output_file):
    """
    Merge multiple WAV files into a single WAV file.
    
    :param audio_files: List of paths to WAV files to merge
    :param output_file: Path to save the merged WAV file
    """
    merged_audio = AudioSegment.empty()
    
    for file_path in audio_files:
        if os.path.exists(file_path):
            audio = AudioSegment.from_wav(file_path)
            merged_audio += audio
    
    # Export the merged audio
    merged_audio.export(output_file, format="wav")

def text_to_speech(input_file, output_file, language="de", speaker=None, model_name=DEFAULT_MODEL, tts_model=None):
    """
    Converts text from an input file to speech and saves it as a WAV file.
    Supports mixed language content using language tags.

    :param input_file: Path to the text file containing input text.
    :param output_file: Path to save the output WAV file.
    :param language: Default language code (e.g., "de" for German).
    :param speaker: Speaker identifier. If None, uses the first available speaker.
    :param model_name: Name of the TTS model to use
    :param tts_model: Pre-loaded TTS model instance
    :return: The TTS model instance
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")

    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read().strip()

    if not text:
        raise ValueError("Input file is empty.")

    # Use the provided model or initialize a new one
    if tts_model is None:
        device = get_device()
        print(f"Using device: {device} with model: {model_name}")
        
        # Initialize TTS and move to device
        tts_model = TTS(model_name=model_name).to(device)
    
    # If no speaker is provided, use the first available one
    if speaker is None or speaker == "random":
        speakers = []
        if hasattr(tts_model.synthesizer.tts_model, 'speaker_manager') and hasattr(tts_model.synthesizer.tts_model.speaker_manager, 'speakers'):
            speakers = tts_model.synthesizer.tts_model.speaker_manager.speakers
        
        if speakers:
            speaker = speakers[0]
            print(f"Using default speaker: {speaker}")
        else:
            raise ValueError("No speakers available in the model")

    # Parse the mixed language text
    segments = parse_mixed_language_text(text)
    
    # Create a temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    try:
        for i, (content, lang, pause_duration) in enumerate(segments):
            if pause_duration > 0:
                # Create a silence file for pause
                pause_file = os.path.join(temp_dir, f"pause_{i}.wav")
                create_silence(pause_duration, pause_file)
                temp_files.append(pause_file)
            elif content:
                # Generate speech for content
                segment_lang = lang if lang else language
                segment_file = os.path.join(temp_dir, f"segment_{i}.wav")
                
                # Generate speech
                tts_model.tts_to_file(
                    text=content,
                    file_path=segment_file,
                    language=segment_lang,
                    speaker=speaker
                )
                temp_files.append(segment_file)
        
        # Merge all the audio files
        if temp_files:
            merge_audio_files(temp_files, output_file)
            print(f"Mixed language audio has been saved to {output_file}")
        else:
            raise ValueError("No audio segments were generated")
            
    finally:
        # Clean up temporary files
        for file_path in temp_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
    
    return tts_model

def text_string_to_speech(text, output_file, language="de", speaker=None, tts_model=None):
    """
    Converts a text string to speech and saves it as a WAV file.
    Supports mixed language content using language tags.

    :param text: String containing input text
    :param output_file: Path to save the output WAV file
    :param language: Default language code (e.g., "de" for German)
    :param speaker: Speaker identifier. If None, uses the first available speaker
    :param tts_model: Pre-loaded TTS model instance
    :return: The TTS model instance
    """
    if not text:
        raise ValueError("Input text is empty.")

    # Use the provided model or initialize a new one
    if tts_model is None:
        device = get_device()
        print(f"Using device: {device}")
        
        # Initialize TTS and move to device
        tts_model = TTS(model_name=DEFAULT_MODEL).to(device)
    
    # If no speaker is provided, use the first available one
    if speaker is None or speaker == "random":
        speakers = []
        if hasattr(tts_model.synthesizer.tts_model, 'speaker_manager') and hasattr(tts_model.synthesizer.tts_model.speaker_manager, 'speakers'):
            speakers = tts_model.synthesizer.tts_model.speaker_manager.speakers
        
        if speakers:
            speaker = speakers[0]
            print(f"Using speaker: {speaker}")
        else:
            raise ValueError("No speakers available in the model")
    
    # Parse the mixed language text
    segments = parse_mixed_language_text(text)

    print(dumps (segments, indent=2))
   
    
    # Create a temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    try:
        for i, (content, lang, pause_duration) in enumerate(segments):
            if pause_duration > 0:
                # Create a silence file for pause
                pause_file = os.path.join(temp_dir, f"pause_{i}.wav")
                create_silence(pause_duration, pause_file)
                temp_files.append(pause_file)
            elif content:
                # Generate speech for content
                segment_lang = lang if lang else language
                segment_file = os.path.join(temp_dir, f"segment_{i}.wav")
                
                # Generate speech
                tts_model.tts_to_file(
                    text=content,
                    file_path=segment_file,
                    language=segment_lang,
                    speaker=speaker
                )
                temp_files.append(segment_file)
        
        # Merge all the audio files
        if temp_files:
            merge_audio_files(temp_files, output_file)
            print(f"Mixed language audio has been saved to {output_file}")
        else:
            raise ValueError("No audio segments were generated")
            
    finally:
        # Clean up temporary files
        for file_path in temp_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
    
    return tts_model

def interactive_mode(language="de", speaker=None, model_name=DEFAULT_MODEL):
    """
    Run in interactive mode, asking for text input and generating speech.
    Supports mixed language input with tags.
    
    :param language: Default language code (e.g., "de" for German)
    :param speaker: Speaker identifier. If None, uses the first available speaker
    :param model_name: Name of the TTS model to use
    """
    print("Loading TTS model (this might take a while)...")
    device = get_device()
    tts_model = TTS(model_name=model_name).to(device)
    print(f"Model loaded on {device}")
    
    # If no speaker is specified, determine the default one
    if speaker is None or speaker == "random":
        speakers = []
        if hasattr(tts_model.synthesizer.tts_model, 'speaker_manager') and hasattr(tts_model.synthesizer.tts_model.speaker_manager, 'speakers'):
            speakers = tts_model.synthesizer.tts_model.speaker_manager.speakers
        
        if speakers:
            speaker = speakers[0]
            print(f"Using default speaker: {speaker}")
        else:
            print("No specific speakers available in this model")
    else:
        print(f"Using speaker: {speaker}")
    
    print("\n=== Interactive TTS Mode ===")
    print("Type your text and press Enter to generate speech.")
    print("Use <en>English text</en> or <de>German text</de> for mixed language.")
    print("Use {PAUSE=400} for a 400ms pause.")
    print("Type 'exit', 'quit', or press Ctrl+C to exit.")
    
    while True:
        try:
            user_input = input("\nEnter text (or 'exit' to quit): ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            if not user_input.strip():
                print("Please enter some text.")
                continue
            
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"output_{timestamp}.wav"
            
            # Generate speech from the mixed language text
            text_string_to_speech(
                user_input,
                output_file,
                language,
                speaker,
                tts_model
            )
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Interactive session ended.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert text to speech using Coqui TTS.")
    parser.add_argument("input_file", type=str, nargs='?', help="Path to the input text file.")
    parser.add_argument("output_file", type=str, nargs='?', help="Path to save the output WAV file.")
    parser.add_argument("--language", type=str, default="de", help="Default language code (default: de for German)")
    parser.add_argument("--speaker", type=str, help="Speaker identifier")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"TTS model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--list_speakers", action="store_true", help="List all available speakers")
    parser.add_argument("--device", action="store_true", help="Show which device (CPU/CUDA) will be used")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode, asking for text input")

    args = parser.parse_args()

    if args.device:
        print(f"Using device: {get_device()}")
    elif args.list_speakers:
        list_available_speakers(args.model)
    elif args.interactive:
        interactive_mode(args.language, args.speaker, args.model)
    elif args.input_file and args.output_file:
        try:
            text_to_speech(
                args.input_file, 
                args.output_file, 
                args.language, 
                args.speaker, 
                args.model
            )
        except Exception as e:
            print(f"Error: {e}")
    else:
        parser.print_help()
