import time
import base64
import glob
import random
from anthropic import Anthropic
from dotenv import load_dotenv
import os
load_dotenv()

random.seed(1337)
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


img_paths = sorted(glob.glob("data/test-img-*.jpg"))
img_paths = img_paths * 3  # Triple the length
random.shuffle(img_paths)  # Shuffle the order
img_base64_strings = [encode_image(img_path) for img_path in img_paths]

questions = [
    "What do you see in this image?",
    "Describe the content of this image concisely.",
    "What's in the image?",
    "Where is this image taken?",
    "Summarize the content of this image in one sentence.",
    "What's the main subject of this image?",
    "What's the background of this image?",
    "What's the color of the background?",
    "What's the texture of the background?",
    "How many objects can you count in this image?",
    "What emotions or mood does this image convey?",
    "What time of day do you think this photo was taken?",
    "Are there any people visible in this image?",
    "What style or genre would you classify this image as?",
    "What details can you notice about the lighting?",
    "What is the meaning of this image?",
]

questions = questions * 3  # Triple the length
random.shuffle(questions)  # Shuffle the order

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_base64_string,
                },
            },
            {"type": "text", "text": question},
        ],
    }
    for img_base64_string, question in zip(img_base64_strings, questions)
]


conversation_history = []

start_time = time.time()
ttft = None

for i, message in enumerate(messages):
    conversation_history.append(message)

    ttft_start_time = time.time()
    stream = client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=conversation_history,
        stream=True,
        max_tokens=16,
    )

    assistant_response = ""
    ttft = None
    for event in stream:
        if ttft is None:
            ttft = time.time() - ttft_start_time
            print(f"Turn {i+1} TTFT: {ttft:.3f}s")

        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            assistant_response += event.delta.text

    print(f"Turn {i+1}: {assistant_response}")
    conversation_history.append({"role": "assistant", "content": assistant_response})

end_time = time.time()
print(f"Time to completion: {end_time - start_time} seconds")
