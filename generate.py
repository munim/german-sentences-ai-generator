from groq import Groq
from tinydb import TinyDB, Query

client = Groq()
completion = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {
            "role": "user",
            "content": "write 5 sentences in A1, 5 sentences in A2 for each in german for the words, also add english translation of each sentences:\nder Einfluss, die Methode. Write output in raw json"
        }
    ],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    response_format={"type": "json_object"},
    stream=False,
    stop=None,
)

print(completion.choices[0].message)

