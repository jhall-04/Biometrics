import numpy as np
import torch
import json
import os

class JSONFaceStore:
    def __init__(self, path='faces.json'):
        """
        Initialize the JSONFaceStore with a specified storage file.
        
        Args:
            path (str): The path to the JSON file where face embeddings will be stored.
        """
        self.path = path
        self.data = self._load_faces()

    def _load_faces(self):
        """Load faces from the JSON file into memory."""
        if not os.path.exists(self.path):
            return {}
        with open(self.path, 'r') as f:
            return json.load(f)

    def _save_face(self):
        """Save the current face to the JSON file."""
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def save_embedding(self, user_id: str, embedding: torch.Tensor):
        """
        Save or update a user embedding in the JSON storage.
        Args:
            user_id (str): The unique identifier for the user.
            embedding (torch.Tensor): The face embedding to save, expected shape is (512,).
        """
        self.data[user_id] = embedding.tolist()
        self._save_face()

    def get_embedding(self, user_id: str) -> torch.Tensor | None:
        """
        Retrieve a user embedding from the JSON storage.
        
        Args:
            user_id (str): The unique identifier for the user.
        
        Returns:
            torch.Tensor: The face embedding as a tensor, or None if not found.
        """
        vec = self.data.get(user_id)
        if vec is None:
            return None
        return torch.tensor(vec)

    def list_users(self) -> list[str]:
        """
        List all user IDs stored in the JSON storage.
        
        Returns:
            list[str]: A list of user IDs.
        """
        return list(self.data.keys())

    def delete_user(self, user_id: str):
        """
        Delete a user embedding from the JSON storage.
        
        Args:
            user_id (str): The unique identifier for the user to delete.
        """
        if user_id in self.data:
            del self.data[user_id]
            self._save_face()
        else:
            raise KeyError(f"User ID {user_id} not found in storage.")

        def get_all_embeddings(self) -> dict[str, torch.Tensor]:
            """
            Retrieve all user embeddings from the JSON storage.
            
            Returns:
                dict[str, torch.Tensor]: A dictionary mapping user IDs to their embeddings.
            """
            return {user_id: torch.tensor(embedding) for user_id, embedding in self.data.items()}
    