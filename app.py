"""
Main application file for the Style Finder Gradio interface.
"""

import gradio as gr
import pandas as pd
import os
from tempfile import NamedTemporaryFile

# Import local modules
from models.image_processor import ImageProcessor
from models.llm_service import LlamaVisionService
from utils.helpers import get_all_items_for_image, format_alternatives_response, process_response
import config

class StyleFinderApp:
    """
    Main application class that orchestrates the Style Finder workflow.
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the Style Finder application.
        
        Args:
            dataset_path (str): Path to the dataset file
            
        Raises:
            FileNotFoundError: If the dataset file is not found
            ValueError: If the dataset is empty or invalid
        """
        # Load the dataset
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
        self.data = pd.read_pickle(dataset_path)
        if self.data.empty:
            raise ValueError("The loaded dataset is empty")
        
        # Initialize components
        self.image_processor = ImageProcessor(
            image_size=config.IMAGE_SIZE,
            norm_mean=config.NORMALIZATION_MEAN,
            norm_std=config.NORMALIZATION_STD
        )
        
        self.llm_service = LlamaVisionService(
            model_id=config.LLAMA_MODEL_ID,
            project_id=config.PROJECT_ID,
            region=config.REGION
        )

    def process_image(self, image):
        """
        Process a user-uploaded image and generate a fashion response.
        
        Args:
            image: PIL image uploaded through Gradio
                
        Returns:
            str: Formatted response with fashion analysis
        """
        # Save the image temporarily if it's not already a file path
        if not isinstance(image, str):
            temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
            image_path = temp_file.name
            image.save(image_path)
        else:
            image_path = image
        
        # Step 1: Encode the image
        user_encoding = self.image_processor.encode_image(image_path, is_url=False)
        if user_encoding['vector'] is None:
            return "Error: Unable to process the image. Please try another image."
        
        # Step 2: Find the closest match
        closest_row, similarity_score = self.image_processor.find_closest_match(user_encoding['vector'], self.data)
        if closest_row is None:
            return "Error: Unable to find a match. Please try another image."
        
        print(f"Closest match: {closest_row['Item Name']} with similarity score {similarity_score:.2f}")
        
        # Step 3: Get all related items
        all_items = get_all_items_for_image(closest_row['Image URL'], self.data)
        if all_items.empty:
            return "Error: No items found for the matched image."
        
        # Step 4: Generate fashion response
        bot_response = self.llm_service.generate_fashion_response(
            user_image_base64=user_encoding['base64'],
            matched_row=closest_row,
            all_items=all_items,
            similarity_score=similarity_score,
            threshold=config.SIMILARITY_THRESHOLD
        )
        
        # Clean up temporary file
        if not isinstance(image, str):
            try:
                os.unlink(image_path)
            except:
                pass
        
        return process_response(bot_response)


def create_gradio_interface(app):
    """
    Create and configure the Gradio interface.
    
    Args:
        app (StyleFinderApp): Instance of the StyleFinderApp
        
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    # TODO: Create Gradio Blocks interface
    
    # TODO: Add introduction section
    
    # TODO: Add example images section
    
    # TODO: Add example image buttons
    
    # TODO: Add image input, submit button, and status components
    
    # TODO: Add output display component
    
    # TODO: Configure submit button click event handlers
    
    # TODO: Configure example image button event handlers
    
    # TODO: Add information about the application
    
    # TODO: Return the configured interface

if __name__ == "__main__":
    try:
        # Initialize the app with the dataset
        app = StyleFinderApp("swift-style-embeddings.pkl")
        
        # Create the Gradio interface
        demo = create_gradio_interface(app)
        
        # Launch the Gradio interface
        demo.launch(
            server_name="127.0.0.1",  
            server_port=5000,
            share=True  # Set to False if you don't want to create a public link
        )
    except Exception as e:
        print(f"Error starting the application: {str(e)}") 