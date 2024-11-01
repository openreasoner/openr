import os
import threading
from transformers import pipeline

# Set the environment variable for the endpoint
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class LLMService:
    """
    A class to manage a large language model (LLM) service using Hugging Face's transformers library.
    """

    def __init__(self, model_name: str = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-Math-7B-Instruct",
                 device: str = "cuda", max_new_tokens: int = 2048,
                 temperature: float = 0.7, top_k: int = 30, top_p: float = 0.9):
        """
        Initialize the LLMService with model parameters and sampling settings.

        Parameters:
        - model_name (str): Path or Hugging Face hub model name.
        - device (str): Device for computation, e.g., 'cuda' or 'cpu'.
        - max_new_tokens (int): Maximum number of new tokens to generate.
        - temperature (float): Sampling temperature for response generation.
        - top_k (int): Top-K sampling parameter for response diversity.
        - top_p (float): Top-P sampling parameter for response diversity.
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.pipe = None
        self.load_lock = threading.Lock()

    def start_service(self):
        """
        Start the LLM service by loading the model into the pipeline if it's not already loaded.
        Ensures thread-safe loading using a lock.
        """
        with self.load_lock:
            if self.pipe is None:
                print(f"Loading model '{self.model_name}' on device '{self.device}'...")
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model_name,
                    torch_dtype="auto",
                    device_map=self.device
                )
                print("Model loaded successfully.")

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the model based on the provided prompt.

        Parameters:
        - prompt (str): The input prompt to generate a response for.

        Returns:
        - str: The generated response from the model.
        """
        if self.pipe is None:
            raise ValueError("LLM service not started. Please call start_service() first.")

        # Generate response from the model
        messages = [{"role": "user", "content": prompt}]
        response_message = self.pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p
        )[0]["generated_text"]
        return response_message
