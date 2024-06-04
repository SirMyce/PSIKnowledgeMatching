"""
One of the key functionalities of this project is the ability to embed, noise, and invert text. This file contains the functions that allow for this functionality. The functions are tested in the test_embeddings_and_rounding function.

You may wish to change the encoders and correctors to suit your needs.
"""

import vec2text
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

device = "cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu"
encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(device)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
corrector = vec2text.load_pretrained_corrector("gtr-base")

 
def embed_text(text_list) -> torch.Tensor:
    """Embeds a list of texts using a pre-trained model and returns the embeddings."""
    inputs = tokenizer(text_list, return_tensors="pt", max_length=128, padding="max_length", truncation=True).to(device)
    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])  # type: ignore
    return embeddings.cpu()
    
def invert_embedding(embeddings, num_steps=20):
    """Inverts the given embeddings using vec2text"""
    inverted_embeddings = vec2text.invert_embeddings(
        embeddings=embeddings.to(device), corrector=corrector, num_steps=20, sequence_beam_width=2
    )
    return inverted_embeddings

def get_edges(splits, original_range):
    """
    Returns a tensor containing rounding edges for the given number of splits.
    """
    # Magic rounding builder to make rounding edges
    diff = np.abs(original_range[1] - original_range[0])
    edges = torch.tensor(original_range)
    for i in range(splits):
        diff /= 2
        edges = torch.cat([edges, edges - diff])
    return edges.float(), diff

def get_linspace_edges(splits, original_range):
    #return a tensor containing the rounding edges for the given number of splits using np.linspace
    edges = np.linspace(original_range[0], original_range[1], num=splits+2)
    return torch.tensor(edges).float()

def linspace_round(linspace_edges, embeddings):
    #Rounds the given embeddings to the nearest value in the 'linspace_edges' array
    return torch.concat([linspace_edges[(linspace_edges - embedding.view(-1,1)).abs().argmin(dim=1)].float().view(1,-1) for embedding in embeddings])

def fakeround(edges, embeddings):
    """
    Rounds the given embeddings to the nearest value in the 'edges' array.
    """
    return torch.concat([edges[(edges - embedding.view(-1,1)).abs().argmin(dim=1)].float().view(1,-1) for embedding in embeddings])

def gauss_noise(embeddings, diff):
    # Match gauss error to the rounding error 
    # https://en.wikipedia.org/wiki/Half-normal_distribution
    mean_error = diff / 4 
    # For two points diff apart, the largest jump to round is diff/2. Since points are uniformly distributed, the average jump is diff/4
    var = np.sqrt( np.pi / 2) * mean_error
    return embeddings+torch.normal(0,var, embeddings.shape).float()

def uniform_noise(embeddings, diff):
    mean_error = diff / 2 
    return embeddings + (torch.rand(embeddings.shape, dtype=torch.float32)*2 - 1)*mean_error


def test_embeddings_and_rounding():
    """This function will run all the relevant functions and test the error rates are acceptable and all code works."""
    text_to_embed = ['This is an example sentence for LLM embedding and reconstruction.', 'This is another example sentence.', 'For this one Im gonna mentioned the great work Micheal and Shayla are doing']

    embeddings = embed_text(text_to_embed)
    assert embeddings.shape[0] == len(text_to_embed)

    embedding_range = [embeddings.min().item(), embeddings.max().item()]

    # Testing the rounding works
    edges, diff = get_edges(6, embedding_range)
    assert len(edges) == 2**(6 + 1)

    diff_matrix = (edges - edges.view(-1,1))
    nearest_edge_diff = torch.min(torch.abs(diff_matrix[~torch.eye(diff_matrix.shape[0], dtype=bool)]))
    assert np.abs(nearest_edge_diff -  diff) < 0.0001 # The nearest edge should be the diff

    rounded = fakeround(edges, embeddings)
    assert rounded.shape == embeddings.shape
    assert torch.mean(rounded - embeddings) < 0.001 # The average error should be close to zero

    assert np.abs(diff / torch.mean(torch.abs(rounded - embeddings)).item()) - 4 < 0.1  # For two points diff apart, the largest jump to round is diff/2. Since points are uniformly distributed, the average jump is diff/4

    # Testing that noise works
    gauss_noised = gauss_noise(embeddings, diff)

    assert torch.mean(gauss_noised - embeddings) < 0.001 # The mean error should be close to zero

    assert np.abs((diff/4) / torch.mean(torch.abs(gauss_noised - embeddings)).item() - 1) < 0.1 # The mean absolute error should be within 10% of the diff

    # Testing uniform noise
    uniform_noised = uniform_noise(embeddings, diff)
    assert torch.mean(uniform_noised - embeddings) < 0.001 # The mean error should be close to zero

    assert np.abs((diff/4) / torch.mean(torch.abs(uniform_noised - embeddings)).item() - 1) < 0.1 # The mean absolute error should be within 10% of the diff

    # Test that inverting works.
    inverted_embeddings = invert_embedding(embeddings, num_steps=2)
    inverted_gauss_noised = invert_embedding(gauss_noised, num_steps=2)
    inverted_uniform_noised = invert_embedding(uniform_noised, num_steps=2)

    # Check that we can embed the inverted embeddings
    closed_loop = embed_text(inverted_embeddings)

