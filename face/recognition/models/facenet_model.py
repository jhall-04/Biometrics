from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F
import torch

class FaceNetEmbedder:
    def __init__(self, pretrained='vggface2', device=None):
        """
        Initialize the FaceNetEmbedder with a pre-trained model.
        
        Args:
            pretrained (str): The name of the pre-trained model to use.
            device (str or torch.device, optional): The device to run the model on. Defaults to CPU if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = InceptionResnetV1(
            pretrained=pretrained, 
            classify=False,
        ).eval().to(self.device)

    
    def get_embedding(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate an embedding for a given face tensor.
        
        Args:
            face_tensor (torch.Tensor): A tensor representing the face image, expected shape is (1, 3, 160, 160).
        
        Returns:
            torch.Tensor: The embedding vector of shape (1, 512).
        """
        face_tensor = face_tensor.to(self.device)
        with torch.no_grad():
            embedding = self.model(face_tensor)
            normalized = F.normalize(embedding, p=2, dim=1)
        return normalized.squeeze(0)  # Remove batch dimension
