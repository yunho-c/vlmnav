import logging
import os
import torch
import numpy as np
import google.generativeai as genai
import cv2

from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, pipeline
from utils import append_mime_tag, encode_image_b64, resize_image_if_needed


class VLM:
    """
    Base class for a Vision-Language Model (VLM) agent.
    This class should be extended to implement specific VLMs.
    """

    def __init__(self, **kwargs):
        """
        Initializes the VLM agent with optional parameters.
        """
        self.name = "not implemented"

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform inference with the VLM agent, passing images and a text prompt.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        """
        raise NotImplementedError

    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with the VLM, incorporating past context.

        Parameters
        ----------
        history : int
            The number of context steps to keep for inference.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the context state of the VLM agent.
        """
        pass

    def rewind(self):
        """
        Rewind the VLM agent one step by removing the last inference context.
        """
        pass

    def get_spend(self):
        """
        Retrieve the total cost or spend associated with the agent.
        """
        return 0


class GeminiVLM(VLM):
    """
    A specific implementation of a VLM using the Gemini API for image and text inference.
    """

    def __init__(self, model="gemini-1.5-flash", system_instruction=None):
        """
        Initialize the Gemini model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        """
        self.name = model
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        # Configure generation parameters
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 500,
            "response_mime_type": "text/plain",
        }

        self.spend = 0
        self.cost_per_input_token = 0.075 / 1_000_000 if 'flash' in self.name else 1.25 / 1_000_000
        self.cost_per_output_token = 0.3 / 1_000_000 if 'flash' in self.name else 5 / 1_000_000

        # Initialize Gemini model and chat session
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=self.generation_config,
            system_instruction=system_instruction
        )
        self.session = self.model.start_chat(history=[])

    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with the Gemini model.

        Parameters
        ----------
        history : int
            The number of environment steps to keep in context.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        images = [Image.fromarray(image[:, :, :3], mode='RGB') for image in images]
        try:
            response = self.session.send_message([text_prompt] + images)
            self.spend += (response.usage_metadata.prompt_token_count * self.cost_per_input_token +
                           response.usage_metadata.candidates_token_count * self.cost_per_output_token)

            # Manage history length based on the number of past steps to keep
            if history == 0:
                self.session = self.model.start_chat(history=[])
            elif len(self.session.history) > 2 * history:
                self.session.history = self.session.history[-2 * history:]

        except Exception as e:
            logging.error(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"

        return response.text

    def rewind(self):
        """
        Rewind the chat history by one step.
        """
        if len(self.session.history) > 1:
            self.model.rewind()

    def reset(self):
        """
        Reset the chat history.
        """
        self.session = self.model.start_chat(history=[])

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform contextless inference with the Gemini model.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        images = [Image.fromarray(image[:, :, :3], mode='RGB') for image in images]
        try:
            response = self.model.generate_content([text_prompt] + images)
            self.spend += (response.usage_metadata.prompt_token_count * self.cost_per_input_token +
                           response.usage_metadata.candidates_token_count * self.cost_per_output_token)

        except Exception as e:
            logging.error(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"

        return response.text

    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        """
        return self.spend


class OpenAIVLM(VLM):
    """
    An implementation using models served via OpenAI API.
    """

    def __init__(self, model="gpt-4o-latest", system_instruction=None, max_image_res=None):
        """
        Initialize the OpenAI model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        """
        from openai import OpenAI
        self.name = model
        self.client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model = model
        self.history = [] 
        self.max_image_res = max_image_res


    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with the OpenAI model.

        Parameters
        ----------
        history : int
            The number of environment steps to keep in context.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        text_contents = [{
            "type": "text",
            "text": text_prompt
        }]
        image_contents = self._image_contents_from_images(images)
        messages = [{"role": "user", "content": text_contents + image_contents}]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history + messages,
            )
            self.history.append(messages[0]) # append user message
            self.history.append({"role": "assistant", "content": [{"type": "text", "text": response.choices[0].message.content}]}) # append response

            # Manage history length based on the number of past steps to keep
            if len(self.history) > 2 * history:
                self.history = self.history[-2 * history:]

        except Exception as e:
            logging.error(f"OPENAI API ERROR: {e}")
            return "OPENAI API ERROR"

        return response.choices[0].message.content


    def rewind(self):
        """
        Rewind the chat history by one step.
        """
        if len(self.history) > 1:
            self.history = self.history[:-2]

    def reset(self):
        """
        Reset the chat history.
        """
        self.history = []


    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform contextless inference with the Gemini model.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        text_contents = [{
            "type": "text",
            "text": text_prompt
        }]
        image_contents = self._image_contents_from_images(images)
        messages = [{"role": "user", "content": text_contents + image_contents}]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )

        except Exception as e:
            logging.error(f"OPENAI API ERROR: {e}")
            return "OPENAI API ERROR"

        return response.choices[0].message.content


    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        NOTE: not implemented
        """
        return 0


    def _image_contents_from_images(self, images: list[np.ndarray]):
        image_contents = [
            {
                "type": "image_url",
                "image_url":
                {
                    "url": append_mime_tag(encode_image_b64(Image.fromarray(image[:, :, :3], mode='RGB'))) \
                    if self.max_image_res is None else append_mime_tag(encode_image_b64(
                        resize_image_if_needed(Image.fromarray(image[:, :, :3], mode='RGB'), self.max_image_res)))
                }
            }
            for image in images
        ]
        return image_contents


class DepthEstimator:
    """
    A class for depth estimation from images using a pre-trained model.
    """

    def __init__(self):
        """
        Initialize the depth estimation pipeline with the appropriate model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = "Intel/zoedepth-kitti"
        self.pipe = pipeline("depth-estimation", model=checkpoint, device=device)

    def call(self, image: np.array):
        """
        Perform depth estimation on an image.

        Parameters
        ----------
        image : np.array
            An RGB image for depth estimation.
        """
        original_shape = image.shape
        image_rgb = Image.fromarray(image[:, :, :3])
        depth_predictions = self.pipe(image_rgb)['predicted_depth']

        # Resize the depth map back to the original image dimensions
        depth_predictions = depth_predictions.squeeze().cpu().numpy()
        depth_predictions = cv2.resize(depth_predictions, (original_shape[1], original_shape[0]))

        return depth_predictions


class Segmentor:
    """
    A class for semantic segmentation using a pre-trained model.
    """

    def __init__(self):
        """
        Initialize the segmentation model and processor.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic").to(self.device)

        # Get class ids for navigable regions
        id2label = self.model.config.id2label
        self.navigability_class_ids = [id for id, label in id2label.items() if 'floor' in label.lower() or 'rug' in label.lower()]

    def get_navigability_mask(self, im: np.array):
        """
        Generate a navigability mask from an input image.

        Parameters
        ----------
        im : np.array
            An RGB image for generating the navigability mask.
        """
        image = Image.fromarray(im[:, :, :3])
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0].cpu().numpy()

        navigability_mask = np.isin(predicted_semantic_map, self.navigability_class_ids)
        return navigability_mask
