import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMHandler:
    """
    Manages the initialization and interaction with the Large Language Model.
    """
    def __init__(self, model_name="Qwen/Qwen2-1.5B-Instruct"):
        """
        Initializes the tokenizer and model.
        
        Args:
            model_name (str): The Hugging Face model identifier.
        """
        logging.info(f"Initializing LLMHandler with model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.device = self.model.device
            logging.info(f"Model {model_name} loaded successfully on device: {self.device}")
        except Exception as e:
            logging.critical(f"Failed to load model or tokenizer: {e}", exc_info=True)
            raise

    def get_response(self, prompt: str, content: str) -> str:
        """
        Generates a response from the LLM based on a prompt and content.

        Args:
            prompt (str): The instructional prompt for the LLM.
            content (str): The text content to be processed.

        Returns:
            str: The LLM's generated response.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant for text analysis. Follow the user's instructions precisely."},
            {"role": "user", "content": f"{content}\n\n---\n\n{prompt}"}
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=4096  # Reduced from 32768 to a more reasonable default
            )
            
            # Slicing to get only the new tokens
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            return response
        
        except Exception as e:
            logging.error(f"Error during LLM generation: {e}", exc_info=True)
            return "Error: Could not generate response."
