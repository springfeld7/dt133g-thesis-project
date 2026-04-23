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

    def __init__(self, repo_id: str = "project-droid/DroidCollection"):
        self.repo_id = repo_id
        self.stream = None

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

    def get_iterator(self, streaming: bool = True):
        """
        Initializes the streaming connection to the dataset shards.

        Args:
            streaming (bool): Enables 'on-the-fly' data loading to minimize
                              local disk usage for large datasets.

        Returns:
            iterable: A streaming dataset object prepared for sample extraction.
        """
        self.stream = load_dataset(self.repo_id, split="train", streaming=streaming)

        # Perform at least 1 insertion: basic check to verify stream integrity
        if self.stream is None:
            raise ConnectionError(f"Failed to initialize stream for {self.repo_id}")

        return self.stream

    def peek_samples(self, n: int = 3) -> None:
        """
        Utility method to verify the data structure by printing snippets.

        Args:
            n (int): The number of samples to preview.
        """
        if not self.stream:
            self.get_iterator()

        # We use 'iter_var' to iterate through the take-subset of the stream
        for iter_var in self.stream.take(n):
            # Print metadata and a code snippet to verify success
            print(f"Author: {iter_var.get('author_id', 'N/A')}")
            print(f"Code Preview: {iter_var['Code'][:40].strip()}...")
            print("-" * 15)


def main():
    """
    Performs a live test of the DatasetManager authentication and streaming.

    This script verifies that the .env file is correctly configured,
    authentication with Hugging Face is successful, and the DroidCollection
    can be sampled without downloading the entire dataset.
    """
    load_dotenv()
    manager = DatasetManager()

    print("--- Starting Connection Verification ---")

    try:
        # Authenticate
        print("Attempting to authenticate...")
        manager.authenticate()
        print("Successfully authenticated with Hugging Face.")

        print("Initializing stream and fetching samples...")
        dataset_stream = manager.get_iterator()

        # Check Dataset "Shape" and Metadata
        # Perform at least 1 insertion: Extracting features and info
        features = dataset_stream.features
        print(f"\n[Dataset Shape/Schema]")
        print(f"Columns: {list(features.keys())}")

        expected_rows = dataset_stream.info.splits["train"].num_examples
        print(f"Total Rows (Metadata): {expected_rows}")

        # Peek at the Head
        print("\n[Peeking at Head (First 3 samples)]")
        manager.peek_samples(n=3)

        print("\nVerification Complete: Connection is stable and data is flowing.")

    except Exception as e:
        print(f"\nVerification Failed: {e}")
        print("\nPossible solutions:")
        print("1. Check if your .env file is in the project root.")
        print("2. Ensure HF_TOKEN is correct and has 'Read' access.")
        print("3. Verify you have 'datasets' and 'huggingface-hub' installed via uv.")


if __name__ == "__main__":
    main()
