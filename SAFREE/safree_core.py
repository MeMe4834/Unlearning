import torch
import math
from transformers import CLIPTokenizer, CLIPTextModel

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def f_beta(z, btype='sigmoid', upperbound_timestep=10, concept_type='nudity'):
    if "artists-" in concept_type:
        t = 5.5
        k = 3.5
    else:
        t = 5.333
        k = 2.5

    if btype == "tanh":
        _value = math.tanh(k * (10 * z - t))
        output = round(upperbound_timestep / 2. * (_value + 1))
    elif btype == "sigmoid":
        sigmoid_scale = 2.0
        _value = sigmoid(sigmoid_scale * k * (10 * z - t))
        output = round(upperbound_timestep * _value)
    else:
        raise NotImplementedError('btype is incorrect')
    return output

def projection_matrix(E):
    """Returns projection matrix P = E (EᵗE)⁻¹ Eᵗ"""
    return E @ torch.pinverse(E.T @ E) @ E.T

def safree_projection(input_embeddings, p_emb, masked_input_subspace_projection, concept_subspace_projection,
                      alpha=0., max_length=77, logger=None):
    device = input_embeddings.device
    (n_t, dim) = p_emb.shape

    I_m_cs = torch.eye(dim).to(device) - concept_subspace_projection
    dist_vec = I_m_cs @ p_emb.T
    dist_p_emb = torch.norm(dist_vec, dim=0)  # (n_t,)

    # 논문 방식: 잔차 벡터 평균보다 작은 애들을 제거
    means = [torch.mean(torch.cat((dist_p_emb[:i], dist_p_emb[i+1:]))) for i in range(n_t)]
    mean_dist = torch.tensor(means).to(device)
    rm_vector = (dist_p_emb < (1. + alpha) * mean_dist).float()  # 1: keep, 0: remove

    if logger is not None:
        logger.log(f"Among {n_t} tokens, we remove {int(n_t - rm_vector.sum())}.")
    else:
        print(f"Among {n_t} tokens, we remove {int(n_t - rm_vector.sum())}.")

    ones_tensor = torch.ones(max_length).to(device)
    ones_tensor[1:n_t + 1] = rm_vector  # [CLS] 제외하고 마스킹
    ones_tensor = ones_tensor.unsqueeze(1)

    text_e = input_embeddings.squeeze(0)  # (77, dim)
    new_text_e = I_m_cs @ masked_input_subspace_projection @ text_e.T
    new_text_e = new_text_e.T  # (77, dim)

    merged_text_e = torch.where(ones_tensor.bool(), text_e, new_text_e)
    new_embeddings = merged_text_e.unsqueeze(0)  # (1, 77, dim)
    return new_embeddings, rm_vector

def get_safree_modified_embeddings(prompt: str, category: str, tokenizer, text_encoder, alpha=0.0, device='cuda'):

    tokenized_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    print("▶️ Tokenized IDs:", tokenized_ids)
    print("▶️ Token count (with special tokens):", tokenized_ids.shape[1])
    print("▶️ n_real_tokens =", tokenized_ids.shape[1] - 2)

    # 1. Original embeddings
    text_inputs = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_embeddings = text_encoder(**text_inputs).last_hidden_state  # (1, 77, dim)

    # 2. Masked prompt embeddings
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    n_real_tokens = untruncated_ids.shape[1] - 2  # exclude [CLS], [SEP]
    masked_ids = untruncated_ids.repeat(n_real_tokens, 1)
    for i in range(n_real_tokens):
        masked_ids[i, i + 1] = 0  # mask each token
    masked_ids = masked_ids.to(device)
    masked_embeddings = text_encoder(input_ids=masked_ids).pooler_output  # (n_real_tokens, dim)

    # 3. Define negative concept tokens
    if category == "nudity":
        negative_prompt_space = [
            "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
            "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
            "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
            "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification"
        ]
    elif "artists-" in category:
        artist_name = category.split("-")[-1]
        if artist_name == "VanGogh":
            negative_prompt_space = ["Van Gogh"]
        elif artist_name == "KellyMcKernan":
            negative_prompt_space = ["Kelly McKernan"]
        else:
            raise ValueError(f"Unknown artist: {artist_name}")
    else:
        raise ValueError("Category must be 'nudity' or 'artists-*'")

    # 4. Embed negative concept tokens
    neg_inputs = tokenizer(negative_prompt_space, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    neg_inputs = {k: v.to(device) for k, v in neg_inputs.items()}
    neg_embeds = text_encoder(**neg_inputs).pooler_output  # (N, dim)

    # 5. Compute projection matrices
    project_matrix = projection_matrix(neg_embeds.T)
    masked_project_matrix = projection_matrix(masked_embeddings.T)

    # 6. Run SAFREE projection
    safree_embeds, rm_vector = safree_projection(
        text_embeddings,
        masked_embeddings,
        masked_project_matrix,
        project_matrix,
        alpha=alpha,
        logger=None
    )

    return text_embeddings, safree_embeds, rm_vector, project_matrix, neg_embeds, negative_prompt_space  # (1, 77, dim), (1, 77, dim), (n_real_tokens,)
