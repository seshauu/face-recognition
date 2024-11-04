import os
import multiprocessing
import torch
import logging
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure CUDA errors are caught at the correct location
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    multiprocessing.freeze_support()

    logging.info("Starting YOLO training script...")

    try:
        # Load the YOLO model
        logging.info("Loading the YOLO model...")
        model = YOLO("runs/detect/train4/weights/best.pt")  # Consider using a larger model
        logging.info("Model loaded successfully.")

        # Define the data configuration
        data_config = "data.yaml"
        logging.info(f"Data configuration set to {data_config}.")

        # Set the device to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # Hyperparameter tuning and training
        logging.info("Starting training...")
        model.train(
            data=data_config,
            epochs=20,  # Further increase number of epochs
            batch=16,  # Batch size
            imgsz=640,  # Image size
            lr0=0.001,  # Lower initial learning rate
            device=device,
            augment=True,  # Data augmentation
            patience=10,  # Early stopping patience
            optimizer='Adam',  # Use Adam optimizer for potentially better performance
            label_smoothing=0.1,  # Apply label smoothing
            rect=True,  # Rectangular training
        )

        logging.info("Training completed successfully.")

    except IndexError as e:
        logging.error(f"IndexError: {e}")
    except RuntimeError as e:
        logging.error(f"RuntimeError: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
