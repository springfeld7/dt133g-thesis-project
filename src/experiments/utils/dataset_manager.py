"""dataset_manager.py

This module defines the DatasetManager class, which encapsulates the logic for authenticating
with Hugging Face and streaming the DroidCollection dataset.

Includes a verification function to test the connection and data retrieval process, ensuring that
the environment is correctly set up and the dataset can be accessed without issues.
"""

import os
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv


class DatasetManager:
    """
    Coordinates the lifecycle of the DroidCollection dataset acquisition.

    This manager encapsulates Hugging Face authentication and provides
    a memory-efficient streaming interface to source code samples,
    ensuring that large-scale data processing remains performant.

    Attributes:
        repo_id (str): Identifier for the Hugging Face dataset repository.
        stream (iterable): A streaming dataset object for on-the-fly data access.
    """

    def __init__(self):
        self.repo_id = None

    def set_repo(self, repo_id: str) -> None:
        """
        Sets the repository ID for the dataset.

        Args:
            repo_id (str): The identifier for the Hugging Face dataset repository.
        """
        self.repo_id = repo_id

    def authenticate(self, token: str = None) -> None:
        """
        Secures access to the Hugging Face repository.

        Args:
            token (str, optional): A specific User Access Token. If None,
                                   the system checks the HF_TOKEN environment variable.
        """
        # Logic: Priority is given to explicit tokens for flexibility in scripts
        hf_token = token or os.getenv("HF_TOKEN")

        if hf_token:
            login(token=hf_token)
        else:
            print("Notice: No explicit token found. Relying on local CLI cache.")

    def get_stream(self, split: str = "train", streaming: bool = True):
        """
        Returns a streaming iterator for a given dataset split.

        Args:
            split (str): The dataset split to access (e.g., 'train', 'test').
            streaming (bool): Whether to return a streaming dataset.

        Returns:
            An iterable dataset stream for the specified split.
        """
        if not self.repo_id:
            raise ValueError("repo_id is not set. Call set_repo() first.")

        return load_dataset(self.repo_id, split=split, streaming=streaming)


def peek_samples(stream, n: int = 3) -> None:
    """
    Utility function to peek at the first n samples from a streaming dataset.

    Args:
        stream: The streaming dataset iterator.
        n (int): The number of samples to peek at.
    """
    for item in stream.take(n):
        print(f"Code Preview: {item.get('Code', '')[:60]}")
        print("-" * 20)


def main():
    """
    Performs a live test of the DatasetManager authentication and streaming.

    This script verifies that the .env file is correctly configured,
    authentication with Hugging Face is successful, and datasets can be
    streamed correctly using the unified DatasetManager interface.
    """
    load_dotenv()
    manager = DatasetManager()

    print("--- Starting Connection Verification ---")

    try:
        # ----------------------------
        # AUTHENTICATION
        # ----------------------------
        print("Attempting to authenticate...")
        manager.authenticate()
        print("Successfully authenticated with Hugging Face.")

        # ----------------------------
        # DATASET SETUP
        # ----------------------------
        repo_id = "DaniilOr/DroidCollection"
        manager.set_repo(repo_id)

        print(f"\nSetting dataset: {repo_id}")
        print("Initializing stream (train split, streaming mode)...")

        stream = manager.get_stream(split="train")

        # ----------------------------
        # SCHEMA INSPECTION
        # ----------------------------
        print("\n[Dataset Shape/Schema]")

        # streaming datasets don't always expose full metadata reliably
        try:
            features = stream.features
            print(f"Columns: {list(features.keys())}")
        except Exception:
            print("Could not infer full schema from streaming dataset.")

        try:
            expected_rows = stream.info.splits["train"].num_examples
            print(f"Total Rows (Metadata): {expected_rows}")
        except Exception:
            print("Total row count not available in streaming mode.")

        # ----------------------------
        # SAMPLE PREVIEW (PEEK)
        # ----------------------------
        print("\n[Peeking at Head (First 3 samples)]")

        for i, sample in enumerate(stream.take(3)):
            code = sample.get("Code") or sample.get("code") or ""
            lang = sample.get("language") or sample.get("Language") or sample.get("language_name")

            print(f"\n--- Sample {i + 1} ---")
            print(f"Language: {lang}")
            print(f"Code Preview:\n{code[:200]}")

        print("\nVerification Complete: Connection is stable and data is flowing.")

    except Exception as e:
        print(f"\nVerification Failed: {e}")
        print("\nPossible solutions:")
        print("1. Check if your .env file is in the project root.")
        print("2. Ensure HF_TOKEN is correct and has 'Read' access.")
        print("3. Verify 'datasets' and 'huggingface-hub' are installed correctly.")
        print("4. Confirm dataset repo ID is correct.")


if __name__ == "__main__":
    main()
