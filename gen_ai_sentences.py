from tinydb import TinyDB, Query
from groq import Groq
import json
import itertools
from jsonschema import validate
import time
import logging
import sys
from logging.handlers import TimedRotatingFileHandler


def createLogger(__name__):
    # Create a logger
    logger = logging.getLogger(__name__)


    # Create a formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')


    # Create a timed rotating file handler
    file_handler = TimedRotatingFileHandler('./logs/log.txt', when='D', interval=1, backupCount=30)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)
    return logger

logger = createLogger(__name__)

def load_json(file):
    d = None
    with open(file) as f:
        d = json.load(f)
    return d


def generate_sentences(table_name):
    db = TinyDB("./data/db.json")
    gen_db = TinyDB("./data/gendb.json")
    ai_obj = None
    # ai_gen_json_obj = load_json('./data/dummy_ai_resp.json')
    words = get_next_words(table_name, db, 2)

    while len(words) > 0:
        try:
            if ai_obj == None:
                logger.info(f"Creating groq request for: {', '.join(words)}")
                ai_obj = get_ai_response(words)

            if ai_obj == None:
                raise Exception("Failed to get AI response")

            for w in ai_obj:
                gen_db.remove(Query()['word_de'] == w['word_de'])
                logger.info(f"Updating status in DB for {w['word_de']}")
                db.table(table_name).update({ 'updated': True }, Query()['word_de'] == w['word_de'])

            logger.info(f"Inserting data into gen_db")
            gen_db.insert_multiple(ai_obj)

        except Exception as e:
            logger.error(e) 
            for w in words:
                logger.info(f"Updating reverting status")
                gen_db.remove(Query()['word_de'] == w)
                db.table(table_name).update({ 'updated': False }, Query()['word_de'] == w)
        
        logger.info(f"Sleeping for 10 secs")
        time.sleep(10)

        words = get_next_words(table_name, db, 2)
        ai_obj = None

def get_next_words(table_name, db, count):
    not_updated_rows = db.table(table_name).search(Query().updated == False)
    first_two_items = itertools.islice(not_updated_rows, count)
    
    words = list(map(lambda d: d['word_de'], first_two_items))
    return words

def get_ai_response(words):
    ai_resp = None
    try:
        client = Groq()
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "user",
                    "content": f"Act as a German language teacher. Consider these two German words: ({', '.join(words)}). Now write 5 senstences in A1 and 5 sentences in A2 for each of the words. add english translation for each sentences. Write output in json format. German sentence attribute should be called 'de' and english sentences should be called 'en'"
                    # "content": f"write 5 sentences in A1, 5 sentences in A2 for each in german for the words ({','.join(words)}), also add english translation of each sentences. Write output in raw json"
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            response_format={"type": "json_object"},
            stream=False,
            stop=None,
        )

        ai_resp = json.loads(completion.choices[0].message.content)
    except Exception as e:
        try:
            as_json = json.loads(e.response.content)
            ai_resp = json.loads(as_json['error']['failed_generation'])

        except Exception as ei:
            logger.error(f"Unable to process failed response. {ei}")
            return None

    return convert_ai_response(ai_resp)

def convert_ai_response(ai_response):
    resp_schema = load_json('./schema/response_array_schema.json')
    try:
        obj = [{"word_de": k, **v} for k, v in ai_response.items()]

        logger.info("Validating response against schema")
        validate(instance=obj, schema=resp_schema)
        return obj
    except:
        logger.error("JSON schema validation failed.")
        return None


generate_sentences("nouns")


