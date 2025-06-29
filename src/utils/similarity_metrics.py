import torch

def cosine_similarity(avg_emb1, avg_emb2):
    '''
    Measures the angle between two vectors.
    Ranges from -1 to 1.
    1 indicates exact same direction, 0 means orthogonal, -1 means opposite direction.
    '''
    cosi = torch.nn.CosineSimilarity(dim=0)
    output = cosi(avg_emb1, avg_emb2)
    return output

def pairwise_similarity(avg_emb1, avg_emb2):
    '''
    Computes pairwise Euclidean distance between corresponding vectors.
    Lower values indicate more similarity.
    '''
    pairwise_dist = torch.nn.PairwiseDistance(p=2)
    output = pairwise_dist(avg_emb1, avg_emb2)
    return output.mean()

def L1_Loss(avg_emb1, avg_emb2):
    '''
    Mean Absolute Error.
    Lower values indicate more similarity.
    '''
    l1_loss = torch.nn.L1Loss()
    output = l1_loss(avg_emb1, avg_emb2)
    return output

def MSE_Loss(avg_emb1, avg_emb2):
    '''
    Mean Squared Error.
    Lower values indicate more similarity.
    '''
    mse = torch.nn.MSELoss()
    output = mse(avg_emb1, avg_emb2)
    return output

def euclidean_distance(emb1, emb2):
    '''
    Euclidean distance between two embeddings.
    Lower values indicate more similarity.
    '''
    euclidean = torch.sqrt(torch.sum((emb1 - emb2).pow(2)))
    return euclidean

