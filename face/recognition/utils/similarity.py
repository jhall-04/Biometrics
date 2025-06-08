from scipy.spatial.distance import cosine

def verify_embedding(embedding1, embedding2, threshold=0.2):
    """
    Verify if two face embeddings are similar enough to be considered the same person.

    Args:
        embedding1 (list): The first face embedding.
        embedding2 (list): The second face embedding.
        threshold (float): The similarity threshold for verification.

    Returns:
        bool: True if the embeddings are similar, False otherwise.
    """
    embedding1 = embedding1.to('cpu').numpy() if hasattr(embedding1, 'to') else embedding1
    embedding2 = embedding2.to('cpu').numpy() if hasattr(embedding2, 'to') else embedding2
    
    # Calculate the cosine distance between the two embeddings
    distance = cosine(embedding1, embedding2)

    # Check if the distance is below the threshold
    return distance < threshold