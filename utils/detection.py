from facenet_pytorch import MTCNN
from PIL import Image
import torch

def detect_face(image: Image.Image, min_confidence: float = 0.95) -> torch.Tensor:
    """
    Detect a face in an image using MTCNN.
    
    Args:
        image (PIL.Image): The input image containing the face to detect.
        
    Returns:
        torch.tensor: A tensor representing the detected face, or None if no face is detected.
    """
    # Initialize MTCNN detector
    detector = MTCNN(keep_all=False, post_process=True, image_size=160)
    
    # Detect faces in the image
    face, prob = detector(image, return_prob=True)
    
    if face is None or prob < min_confidence:
        return None  # No face detected
    
    # MTCNN returns a tensor of shape (3, 160, 160)
    return face.unsqueeze(0)  # Add batch dimension