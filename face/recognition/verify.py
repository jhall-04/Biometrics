from face.recognition.models.facenet_model import FaceNetEmbedder
from face.recognition.storage.json_storage import JSONFaceStore
from utils.detection import detect_face
from face.recognition.utils.similarity import verify_embedding

from PIL import Image

def verify_face(image: Image.Image, id: str):
    """
    Verify a face from an image with the face recognition system.
    
    Args:
        image (PIL image): The input image containing the face to verify.
        id (str, optional): The identifier for the face.
        
    Returns:
        str: The ID of the enrolled face.
    """
    # Generate embedding for the detected face
    embedder = FaceNetEmbedder()
    embedding = embedder.get_embedding(image)

    # Load the stored embedding for the given ID
    store = JSONFaceStore()
    stored_embedding = store.get_embedding(id)
    
    if stored_embedding is None:
        raise ValueError(f"No embedding found for ID: {id}")
    
    # Verify the embedding against the stored one
    is_verified = verify_embedding(embedding, stored_embedding)
    
    return is_verified