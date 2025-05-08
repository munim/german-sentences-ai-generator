import json
import argparse

def convert_json_unicode(input_file_path, output_file_path):
    """
    Reads a JSON file, decodes Unicode escape sequences in its string literals,
    and writes the modified JSON to an output file with actual Unicode characters.

    Args:
        input_file_path (str): Path to the input JSON file.
        output_file_path (str): Path to the output JSON file.
    """
    try:
        # Read the input JSON file.
        # json.load() automatically decodes \uXXXX sequences into Python Unicode strings.
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        # Write the Python object to the output JSON file.
        # ensure_ascii=False ensures that Unicode characters are written as-is (e.g., 'ö' instead of '\u00f6').
        # indent=2 is for pretty-printing the output JSON.
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=2)

        print(f"Successfully converted '{input_file_path}' to '{output_file_path}'. Unicode characters are now represented directly.")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{input_file_path}'. It might not be a valid JSON file. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert a JSON file with Unicode escape sequences (e.g., \\u00f6) to one with actual Unicode characters (e.g., ö)."
    )
    parser.add_argument("input_file", help="Path to the input JSON file.")
    parser.add_argument("output_file", help="Path to the output JSON file.")

    args = parser.parse_args()

    convert_json_unicode(args.input_file, args.output_file)