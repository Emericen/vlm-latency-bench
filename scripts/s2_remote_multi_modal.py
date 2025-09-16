import os
import time
import base64
import glob
import random
import argparse
import copy
from anthropic import Anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_user_messages(repeat=3, seed=1337):
    """Returns a random list of user questions about some 720p images"""
    random.seed(seed)
    img_paths = sorted(glob.glob("data/test-img-*.jpg"))
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
        "How would you name this image?",
        "What do you find most interesting about this image?",
        "What is the message of this image?",
        "What is the moral of this image?",
        "What did you learn from this image about B2B SaaS?",
    ]

    img_paths = img_paths * repeat
    questions = questions * repeat

    random.shuffle(img_paths)
    random.shuffle(questions)

    user_messages = []
    for image, question in zip(img_paths, questions):
        img_b64 = encode_image(image)
        user_messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": question},
                ],
            }
        )
    return user_messages


def wrap_prompt_caching_signature(messages: list[dict]):
    """
    For fairness, we use Anthropic's prompt caching to (ideally) speed up inference and save token money.

    Official documentation: https://docs.claude.com/en/docs/build-with-claude/prompt-caching
    """
    messages_copy = copy.deepcopy(messages)
    final_message = messages_copy[-1]
    final_message_content_dict = final_message["content"][-1]
    final_message_content_dict["cache_control"] = {"type": "ephemeral"}
    return messages_copy


def run_conversation(
    client: Anthropic,
    model_name: str,
    max_tokens: int,
    repeat: int,
    seed: int,
):
    """have LLM answer each user message in one multi-turn conversation and record time to first token total time to completion on each turn"""
    user_messages = load_user_messages(repeat=repeat, seed=seed)

    # stats storage
    times_to_first_token = []
    times_to_completion = []
    assistant_responses = []

    chat_history = []
    for i, user_message in enumerate(user_messages):
        chat_history.append(user_message)
        start_time = time.time()
        chat_history_w_prompt_caching = wrap_prompt_caching_signature(chat_history)
        try:
            with client.messages.stream(
                messages=chat_history_w_prompt_caching,
                model=model_name,
                max_tokens=max_tokens,
                temperature=0.0,
            ) as stream:
                ttft = None
                response_tokens = []
                for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        response_tokens.append(event.delta.text)
                        ttft = time.time() - start_time if ttft is None else ttft

                # Get usage info from final message
                final_message = stream.get_final_message()
                usage = final_message.usage
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
                cache_creation = getattr(usage, "cache_creation_input_tokens", 0)
                cache_read = getattr(usage, "cache_read_input_tokens", 0)
                print(f"Turn {i+1} Input Tokens: {input_tokens}")
                print(f"Turn {i+1} Output Tokens: {output_tokens}")
                print(f"Turn {i+1} Cache Creation Tokens: {cache_creation}")
                print(f"Turn {i+1} Cache Read Tokens: {cache_read}")

        except KeyboardInterrupt:
            print("Interrupted by user")
            break

        assistant_response = "".join(response_tokens)
        end_time = time.time() - start_time

        print(f"Turn {i+1} TTFT: {ttft:.3f}s")
        print(f"Turn {i+1} Time to completion: {end_time:.3f}s")
        print(f"Turn {i+1} Response: {assistant_response}")
        print()

        times_to_first_token.append(ttft)
        times_to_completion.append(end_time)
        assistant_responses.append(assistant_response)
        chat_history.append({"role": "assistant", "content": assistant_response})

    return times_to_first_token, times_to_completion, assistant_responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark local model multi-modal inference latency."
    )
    parser.add_argument("--model_name", default="claude-sonnet-4-20250514")
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--data_repeat", type=int, default=3)
    parser.add_argument("--data_seed", type=int, default=1337)
    args = parser.parse_args()

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    ttft, ttc, responses = run_conversation(
        client=client,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        repeat=args.data_repeat,
        seed=args.data_seed,
    )

    df = pd.DataFrame(
        {
            "times_to_first_token": ttft,
            "times_to_completion": ttc,
            "assistant_responses": responses,
        }
    )
    df.to_csv(args.output_file, index=False)
