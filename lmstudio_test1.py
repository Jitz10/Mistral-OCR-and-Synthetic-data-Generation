# Example: reuse your existing OpenAI client code
# Change the baseUrl to point to LM Studio
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1247/v1",
                api_key="lm-studio")

completion = client.chat.completions.create(
  model="gemma-3-4b",
  messages=[
    {"role": "system", "content": "Always answer in rhymes."},
    {"role": "user", "content": "Introduce yourself."}
  ],
  temperature=0.7,
)

print(completion.choices[0].message)
