import json
import sys

def generate_tts_text(json_filepath):
    """
    Reads a JSON file containing German verb data and generates text-to-speech ready output.

    Args:
        json_filepath (str): The path to the input JSON file.

    Returns:
        str: A string containing the formatted text for text-to-speech,
             with each verb's text on a new line, designed for more natural
             pronunciation with added pauses.
    """
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return f"Error: File not found at {json_filepath}"
    except json.JSONDecodeError:
        return f"Error: Could not decode JSON from {json_filepath}"
    except Exception as e:
        return f"An unexpected error occurred while reading the file: {e}"

    if not isinstance(data, list):
        return "Error: JSON content is not a list of verb objects."

    output_texts = []
    for verb_data in data:
        if not isinstance(verb_data, dict):
            print(f"Warning: Skipping non-object item in JSON data: {verb_data}")
            continue

        de_data = verb_data.get("de", {})
        en_data = verb_data.get("en", {})

        german_verb = de_data.get("verb", "")
        english_verb = en_data.get("verb", "")
        past_tense = de_data.get("past_tense", "")
        past_participle = de_data.get("past_participle", "")
        sentences = de_data.get("sentences", {})
        present_sentence = sentences.get("present", "")
        past_sentence = sentences.get("past", "")
        past_participle_sentence = sentences.get("past_participle", "")

        # Format the text for more natural text-to-speech with added pauses (commas and periods)
        tts_text = (
            f"{german_verb}, "
            f"{english_verb}. "
            f"{past_tense}, "
            f"{past_participle}. "
            f"{present_sentence}{'. ' if present_sentence and not present_sentence.strip().endswith(('.', '!', '?')) else ' '}"
            f"{past_sentence}{'. ' if past_sentence and not past_sentence.strip().endswith(('.', '!', '?')) else ' '}"
            f"{past_participle_sentence}{'.' if past_participle_sentence and not past_participle_sentence.strip().endswith(('.', '!', '?')) else ''}"
        )
        output_texts.append(tts_text)

    return "\n\n".join(output_texts)

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python generate_tts_text.py <input_json_filepath> [output_filepath]")
        sys.exit(1)

    input_filepath = sys.argv[1]
    tts_output = generate_tts_text(input_filepath)

    if len(sys.argv) == 3:
        output_filepath = sys.argv[2]
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(tts_output)
            print(f"Successfully generated TTS text to {output_filepath}")
        except IOError as e:
            print(f"Error writing to file {output_filepath}: {e}")
            sys.exit(1)
    else:
        print(tts_output)