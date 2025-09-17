import time
import base64
import glob
import random
import argparse
from openai import OpenAI
import pandas as pd


def load_user_messages(repeat=3, seed=1337):
    """Returns a random list of user questions about some 1000+ token texts"""
    random.seed(seed)
    txt_paths = sorted(glob.glob("data/test-txt-*.txt"))
    texts = []
    for txt_path in txt_paths:
        with open(txt_path, "r", encoding="utf-8") as f:
            texts.append(f.read())

    questions = [
        "What is the main topic of this text?",
        "Summarize this text in one sentence.",
        "What is the key message in this text?",
        "What genre or type of writing is this?",
        "What is the tone of this text?",
        "Who is the intended audience for this text?",
        "What emotions does this text convey?",
        "What is the author's main argument?",
        "What evidence does the text provide?",
        "What conclusions can you draw from this text?",
        "What is the most important sentence in this text?",
        "What questions does this text raise?",
        "What is the writing style of this text?",
        "What themes are present in this text?",
        "What is the purpose of this text?",
        "What did you learn from this text?",
        "How would you categorize this text?",
        "What is the central idea of this text?",
        "What perspective does this text represent?",
        "What makes this text compelling or interesting?",
    ]

    texts = texts * repeat
    questions = questions * repeat

    random.shuffle(texts)
    random.shuffle(questions)

    user_messages = []
    for text, question in zip(texts, questions):
        user_messages.append(
            {"role": "user", "content": f"{text}\n\n --- \n\n{question}"}
        )
    return user_messages


def run_conversation(
    client: OpenAI,
    model_name: str,
    max_tokens: int,
    repeat: int,
    seed: int,
):
    """have LLM answer each user message in one multi-turn conversation and record time to completion on each turn"""
    user_messages = load_user_messages(repeat=repeat, seed=seed)

    # stats storage
    times_to_completion = []
    assistant_responses = []

    chat_history = []
    for i, user_message in enumerate(user_messages):
        chat_history.append(user_message)
        start_time = time.time()
        response = client.chat.completions.create(
            messages=chat_history,
            model=model_name,
            max_tokens=max_tokens,
            temperature=0.0,
            stream=False,
        )
        assistant_response = response.choices[0].message.content
        end_time = time.time() - start_time

        print(f"Turn {i+1} Time to completion: {end_time:.3f}s")
        print(f"Turn {i+1} Response: {assistant_response}")
        print()

        times_to_completion.append(end_time)
        assistant_responses.append(assistant_response)
        chat_history.append({"role": "assistant", "content": assistant_response})

    return times_to_completion, assistant_responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark local model multi-modal inference latency."
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--base_url", default="http://192.222.53.224:443/v1")
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--data_repeat", type=int, default=3)
    parser.add_argument("--data_seed", type=int, default=1337)
    parser.add_argument(
        "--output_file", type=str, default="results/qwen2.5_vl_7b_instruct_results.csv"
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    times_to_completion, responses = run_conversation(
        client=client,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        repeat=args.data_repeat,
        seed=args.data_seed,
    )

    df = pd.DataFrame(
        {
            "times_to_completion": times_to_completion,
            "assistant_responses": responses,
        }
    )
    df.to_csv(args.output_file, index=False)
