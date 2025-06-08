from face.recognition.models.facenet_model import FaceNetEmbedder
from face.recognition.storage.json_storage import JSONFaceStore
from face.recognition.utils.detection import detect_face

from PIL import Image
import uuid

def enroll_face(image: Image.Image, id: str=None):
    """
    Enroll a face from an image into the face recognition system.
    
    Args:
        image_path (str): The input image containing the face to enroll.
        id (str, optional): The identifier for the enrolled face. If None, a new ID will be generated.
        
    Returns:
        str: The ID of the enrolled face.
    """
    # Detect faces in the image
    face_tensor = detect_face(image)
    
    if face_tensor is None:
        raise ValueError("No face detected in the image.")
    
    # Generate embedding for the detected face
    embedder = FaceNetEmbedder()
    embedding = embedder.get_embedding(face_tensor)

    # Generate a unique ID if not provided
    if id is None:
        id = str(uuid.uuid4())
    
    # Store the embedding in JSON storage
    store = JSONFaceStore()
    
    store.save_embedding(id, embedding)
    
    return id