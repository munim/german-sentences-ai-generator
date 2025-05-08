#!/usr/bin/env python3
"""
German Verb Sentence Generator using OpenRouter API
This program processes a list of German verbs and generates sentence examples
"""

import os
import json
import time
import random
import argparse
import httpx
import csv
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv


# Configuration class to store settings
class Config:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Check for required OPENROUTER_API_KEY
        if not os.getenv('OPENROUTER_API_KEY'):
            raise ValueError("OPENROUTER_API_KEY not found in environment variables. "
                           "Create a .env file with your API key.")
            
        # Check for required SYSTEM_PROMPT
        if not os.getenv('SYSTEM_PROMPT'):
            raise ValueError("SYSTEM_PROMPT not found in environment variables. "
                           "Add it to your .env file.")
        
        # API settings
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.model = os.getenv('OPENROUTER_MODEL', 'anthropic/claude-3-sonnet')
        self.http_referer = os.getenv('HTTP_REFERER', 'https://github.com/german-verb-generator')
        
        # LLM parameters
        self.system_prompt = os.getenv('SYSTEM_PROMPT')
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('MAX_TOKENS', '4000'))
        
        # Processing settings
        self.batch_size = int(os.getenv('BATCH_SIZE', '50'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.retry_delay = int(os.getenv('RETRY_DELAY', '5000'))
        self.temp_dir = os.getenv('TEMP_DIR', './.temp')


class GermanVerbGenerator:
    def __init__(self, config: Config, input_file: str, output_file: str, prompt_template: str):
        self.config = config
        self.input_file = input_file
        self.output_file = output_file
        self.prompt_template = prompt_template
        self.temp_dir = Path(config.temp_dir)
        
    def run(self) -> None:
        """Main function to process the verbs"""
        try:
            print("Starting German Verb Sentence Generator...")
            
            # Create temp directory if it doesn't exist
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Read input file
            print(f"Reading verbs from {self.input_file}...")
            verb_list = self.read_verbs_from_file(self.input_file)
            print(f"Found {len(verb_list)} verbs to process.")
            
            # Split verbs into batches
            verb_batches = self.chunk_array(verb_list, self.config.batch_size)
            print(f"Split into {len(verb_batches)} batches of max {self.config.batch_size} verbs each.")
            
            # Process each batch
            results = []
            processed_count = 0
            
            for i, verb_batch in enumerate(verb_batches):
                batch_number = i + 1
                print(f"Processing batch {batch_number}/{len(verb_batches)}...")
                
                result = self.process_verb_batch(verb_batch, batch_number)
                
                if result["success"]:
                    results.extend(result["data"])
                    processed_count += len(result["data"])
                    print(f"Batch {batch_number} completed. Processed {processed_count}/{len(verb_list)} verbs.")
                else:
                    print(f"Failed to process batch {batch_number}: {result['error']}")
                    # Save what we have so far to avoid losing progress
                    self.save_results(results)
                
                # Save incremental results
                self.save_results(results)
            
            print(f"Processing complete! Generated sentences for {len(results)} verbs.")
            print(f"Results saved to {self.output_file}")
            
            # Clean up temp directory
            self.cleanup_temp_dir()
            
        except Exception as e:
            print(f"Error in main process: {e}")
            raise
    
    def read_verbs_from_file(self, file_path: str) -> List[Dict[str, str]]:
        """Read verbs from input CSV file"""
        try:
            verbs_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header row
                for row in reader:
                    if len(row) >= 2:
                        german_verb = row[0].strip()
                        english_verb = row[1].strip()
                        if german_verb and english_verb:
                            verbs_data.append({"de": german_verb, "en": english_verb})
            
            if not verbs_data:
                raise ValueError(f"No valid verb data found in {file_path}. Ensure it's a CSV with a header and at least two columns (German, English).")
                
            return verbs_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found at {file_path}")
        except Exception as e:
            print(f"Error reading input CSV file: {e}")
            raise
    
    def process_verb_batch(self, verbs: List[Dict[str, str]], batch_number: int) -> Dict[str, Any]:
        """Process a batch of verbs"""
        temp_file = self.temp_dir / f"batch_{batch_number}.json"
        
        # Check if this batch was already processed
        try:
            if temp_file.exists():
                with open(temp_file, 'r', encoding='utf-8') as f:
                    temp_data = f.read()
                print(f"Found cached result for batch {batch_number}. Skipping API call.")
                return {"success": True, "data": json.loads(temp_data)}
        except Exception:
            # No cached data found or error reading it, proceed with API call
            pass
        
        # Create prompt for the batch
        prompt = self.create_prompt(verbs)
        
        # Call the API with retries
        for attempt in range(1, self.config.max_retries + 1):
            try:
                print(f"Batch {batch_number}, attempt {attempt}/{self.config.max_retries}...")
                
                response = self.call_openrouter_api(prompt)
                parsed_data = self.parse_response(response, verbs)
                
                # Cache the result
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(parsed_data, indent=2))
                
                return {"success": True, "data": parsed_data}
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                
                # Calculate exponential backoff delay
                backoff_delay = self.config.retry_delay * (2 ** (attempt - 1))
                jitter = random.randint(0, 1000)  # Add random jitter
                delay_with_jitter = backoff_delay + jitter
                
                if attempt < self.config.max_retries:
                    print(f"Retrying in {round(delay_with_jitter / 1000)} seconds...")
                    time.sleep(delay_with_jitter / 1000)
                else:
                    return {"success": False, "error": str(e)}
    
    def create_prompt(self, verbs: List[Dict[str, str]]) -> str:
        """Create the prompt for the LLM"""
        # Format the verb list for the prompt, including English translations
        verb_list_formatted = '\n'.join([f"- {v['de']} ({v['en']})" for v in verbs])
        
        try:
            # Read prompt template from file
            with open(self.prompt_template, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # Replace placeholder with formatted verb list
            return template.replace("{{VERB_LIST}}", verb_list_formatted)
        except Exception as e:
            error_msg = f"Failed to read prompt template file: {self.prompt_template}. Please provide a valid prompt template file."
            print(f"Error reading prompt template: {e}")
            raise ValueError(error_msg)
    
    def call_openrouter_api(self, prompt: str) -> httpx.Response:
        """Call the OpenRouter API"""
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.config.api_key}",
                        "HTTP-Referer": self.config.http_referer,
                    },
                    json={
                        "model": self.config.model,
                        "messages": [
                            {"role": "system", "content": self.config.system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                        "stream": False,
                    },
                )
            
            if response.status_code != 200:
                try:
                    error_data = response.json()
                except Exception:
                    error_data = None
                
                raise ValueError(f"API request failed with status {response.status_code}: {error_data}")
            
            print(f"Raw LLM response content: {response.text}")
            
            return response
        except Exception as e:
            raise ValueError(f"OpenRouter API error: {str(e)}")
    
    def parse_response(self, response: httpx.Response, original_verbs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Parse the API response"""
        try:
            # Extract the content from the response
            content = response.json()["choices"][0]["message"]["content"]
            
            # Try to parse the JSON response
            try:
                parsed_data = json.loads(content)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from the response
                import re
                # Updated regex to handle markdown code blocks
                json_match = re.search(r'```json\s*(\[[\s\S]*?])\s*```', content)
                if json_match:
                    # Use the captured group (the JSON array)
                    json_string = json_match.group(1)
                    parsed_data = json.loads(json_string)
                else:
                    # Fallback to original regex if markdown not found
                    json_match = re.search(r'\[[\s\S]*\]', content)
                    if json_match:
                        parsed_data = json.loads(json_match.group(0))
                    else:
                        raise ValueError("Could not extract valid JSON from response")
            
            # Validate that we have an array
            if not isinstance(parsed_data, list):
                raise ValueError("Response is not a JSON array")
            
            # Validate each entry and check against original verb list
            validated_data = []
            processed_verbs = set()
            original_german_verbs = {v['de'] for v in original_verbs}
            
            for entry in parsed_data:
                validated_entry = self.validate_verb_entry(entry)
                if validated_entry:
                    validated_data.append(validated_entry)
                    processed_verbs.add(validated_entry["de"]["verb"])
            
            # Check for missing verbs from the original list
            for verb in original_german_verbs:
                if verb not in processed_verbs:
                    print(f'Verb "{verb}" was not included in the API response.')
            
            return validated_data
        except Exception as e:
            raise ValueError(f"Failed to parse API response: {str(e)}")
    
    def validate_verb_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and fix a verb entry"""
        try:
            # Check basic structure
            if not entry or "de" not in entry or "en" not in entry:
                return None
            
            # Ensure all required fields exist
            validated = {
                "de": {
                    "verb": entry.get("de", {}).get("verb", ""),
                    "infinitive": entry.get("de", {}).get("infinitive") or entry.get("de", {}).get("verb", ""),
                    "type": entry.get("de", {}).get("type", "unknown"),
                    "past_tense": entry.get("de", {}).get("past_tense", ""),
                    "past_participle": entry.get("de", {}).get("past_participle", ""),
                    "sentences": {
                        "present": entry.get("de", {}).get("sentences", {}).get("present") or f"Ich {entry.get('de', {}).get('verb', '')}.",
                        "past": entry.get("de", {}).get("sentences", {}).get("past") or f"Ich habe {entry.get('de', {}).get('verb', '')}."
                    }
                },
                "en": {
                    "verb": entry.get("en", {}).get("verb", ""),
                    "sentences": {
                        "present": entry.get("en", {}).get("sentences", {}).get("present", ""),
                        "past": entry.get("en", {}).get("sentences", {}).get("past", "")
                    }
                }
            }
            
            return validated
        except Exception as e:
            print(f"Error validating entry: {e}, {entry}")
            return None
    
    def save_results(self, results: List[Dict[str, Any]]) -> None:
        """Save results to output file"""
        try:
            # Save results to output file
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(results, indent=2))
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def chunk_array(self, array: List[Any], size: int) -> List[List[Any]]:
        """Split array into chunks"""
        return [array[i:i + size] for i in range(0, len(array), size)]
    
    def cleanup_temp_dir(self) -> None:
        """Clean up temporary directory"""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Failed to clean up temp directory: {e}")


def generate_default_env_file() -> None:
    """Generate default .env file if it doesn't exist"""
    env_path = Path("./.env")
    
    if not env_path.exists():
        default_env = """# OpenRouter API Configuration
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=anthropic/claude-3-sonnet
HTTP-Referer=https://github.com/german-verb-generator

# Processing Configuration
BATCH_SIZE=50
MAX_RETRIES=3
RETRY_DELAY=5000
TEMP_DIR=./.temp

# LLM Parameters
SYSTEM_PROMPT=You are an expert German language teacher with deep knowledge of verb usage, grammar, and idiomatic expressions.
LLM_TEMPERATURE=0.7
MAX_TOKENS=4000
"""
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(default_env)
        print(f"Created default .env file at {env_path}")


def show_help() -> None:
    """Display help information"""
    help_text = """
German Verb Sentence Generator

Usage:
  python german_verb_generator.py --input <input-file> [options]

Options:
  --input, -i     Input CSV file with German and English verbs (required)
  --output, -o    Output JSON file (default: german_verbs_sentences.json)
  --prompt, -p    Prompt template file (required)
  --help, -h      Show this help information

Environment variables (in .env file):
  OPENROUTER_API_KEY    Your OpenRouter API key (required)
  OPENROUTER_MODEL      LLM model to use (default: anthropic/claude-3-sonnet)
  SYSTEM_PROMPT         System prompt for the LLM (required)
  BATCH_SIZE            Number of verbs to process in each batch (default: 50)
  MAX_RETRIES           Maximum retry attempts for API calls (default: 3)
  RETRY_DELAY           Base delay between retries in ms (default: 5000)
  LLM_TEMPERATURE       Temperature parameter for LLM (default: 0.7)
  MAX_TOKENS            Maximum tokens for LLM response (default: 4000)
  """
    print(help_text)


def main():
    """Main entry point for the program"""
    parser = argparse.ArgumentParser(description="German Verb Sentence Generator", add_help=False)
    parser.add_argument("--input", "-i", help="Input CSV file with German and English verbs", required=False)
    parser.add_argument("--output", "-o", help="Output JSON file", default="german_verbs_sentences.json")
    parser.add_argument("--prompt", "-p", help="Prompt template file", required=False)
    parser.add_argument("--help", "-h", action="store_true", help="Show help information")
    
    args = parser.parse_args()
    
    # Show help if requested or if required arguments are missing
    if args.help or not args.input or not args.prompt:
        show_help()
        return
    
    # Initialize environment
    generate_default_env_file()
    
    # Check if prompt template exists
    if not Path(args.prompt).exists():
        print(f"Error: Prompt template file not found at {args.prompt}")
        return
    
    try:
        # Initialize configuration
        config = Config()
        
        # Display current configuration
        print("Current configuration:")
        print(f"- Input file: {args.input}")
        print(f"- Output file: {args.output}")
        print(f"- Prompt template: {args.prompt}")
        print(f"- Model: {config.model}")
        print(f"- Batch size: {config.batch_size}")
        print(f"- System prompt: {config.system_prompt[:50]}...")
        print("")
        
        # Create and run the generator
        generator = GermanVerbGenerator(config, args.input, args.output, args.prompt)
        generator.run()
        
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
