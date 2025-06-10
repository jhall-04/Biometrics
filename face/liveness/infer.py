from face.liveness.model import LivenessModel
from utils.detection import detect_face
from PIL import Image

def infer_liveness(image: Image.Image) -> str:
    """
    Infer liveness from an image using a pre-trained model.
    
    Args:
        image (PIL.Image): The input image containing the face to analyze.
        
    Returns:
        str: 'live' if the face is detected as live, 'spoof' otherwise.
    """
    # Detect face in the image
    face_tensor = detect_face(image)
    
    if face_tensor is None:
        return 'no_face_detected'
    
    # Initialize the liveness model
    model = LivenessModel()
    
    # Perform inference
    result = model.infer(face_tensor)

    if result:
        return face_tensor
    else:
        return None