import time
import glob
import random
import argparse
import uuid
import pandas as pd
from PIL import Image
from vllm import LLM, SamplingParams


def load_images_and_questions(repeat=3, seed=1337):
    """Returns a list of (image, question, uuid) tuples for vLLM native format"""
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

    data = []
    for image_path, question in zip(img_paths, questions):
        # Create stable UUID based on image path for caching
        stable_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, image_path))
        pil_image = Image.open(image_path)
        data.append((pil_image, question, stable_uuid))

    return data


def run_conversation(
    llm: LLM,
    max_tokens: int,
    repeat: int,
    seed: int,
):
    """Run multi-turn conversation with vLLM native format for proper image caching"""
    data = load_images_and_questions(repeat=repeat, seed=seed)

    # stats storage
    times_to_completion = []
    assistant_responses = []

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    # Build conversation history for multi-turn chat
    conversation_history = ""

    for i, (pil_image, question, stable_uuid) in enumerate(data):
        # For first turn, include image data. For subsequent turns, use None for cache hit
        if i == 0:
            multi_modal_data = {"image": pil_image}
        else:
            multi_modal_data = {"image": None}  # Cache hit - this works with native vLLM!

        # Build prompt with conversation history
        if i == 0:
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
        else:
            prompt = f"{conversation_history}USER: <image>\n{question}\nASSISTANT:"

        inputs = {
            "prompt": prompt,
            "multi_modal_data": multi_modal_data,
            "multi_modal_uuids": {"image": stable_uuid},
        }

        start_time = time.time()
        outputs = llm.generate([inputs], sampling_params=sampling_params)
        end_time = time.time() - start_time

        assistant_response = outputs[0].outputs[0].text

        print(f"Turn {i+1} Time to completion: {end_time:.3f}s")
        print(f"Turn {i+1} Response: {assistant_response}")
        print()

        times_to_completion.append(end_time)
        assistant_responses.append(assistant_response)

        # Update conversation history
        conversation_history += f"USER: <image>\n{question}\nASSISTANT: {assistant_response}\n"

    return times_to_completion, assistant_responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark local model multi-modal inference latency with native vLLM."
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--data_repeat", type=int, default=3)
    parser.add_argument("--data_seed", type=int, default=1337)
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/32B_native_vllm_cache_run_1.csv",
    )
    args = parser.parse_args()

    # Initialize vLLM with multimodal processor cache enabled
    llm = LLM(
        model=args.model_name,
        mm_processor_cache_gb=4.0,  # Enable 4GB cache for multimodal processing
    )

    times_to_completion, responses = run_conversation(
        llm=llm,
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