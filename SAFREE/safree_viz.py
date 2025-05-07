import torch
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPTokenizer
from safree_core import get_safree_modified_embeddings  # SAFREE 핵심 함수
from torch.utils.tensorboard import SummaryWriter

def plot_cosine_similarity(original_emb, modified_emb):
    ori = original_emb.squeeze(0)
    mod = modified_emb.squeeze(0)
    cos = F.cosine_similarity(ori, mod, dim=-1).detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(cos, label="Cosine Similarity")
    plt.title("SAFREE Projection Effect")
    plt.xlabel("Token Index")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("cosine_similarity.png")
    print("✅ Cosine similarity plot saved: cosine_similarity.png")


def decode_nearest_tokens(modified_embeds, original_embeds, tokenizer, text_encoder, rm_vector, prompt):
    embeds = modified_embeds.squeeze(0)
    original = original_embeds.squeeze(0)
    embeds_norm = embeds / embeds.norm(dim=-1, keepdim=True)

    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}

    tokenized = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt", add_special_tokens=False)
    token_ids = tokenized.input_ids.squeeze(0)

    token_embeddings = text_encoder.get_input_embeddings().weight
    token_embeddings = token_embeddings / token_embeddings.norm(dim=-1, keepdim=True)

    print(f"\n원본 프롬프트: {prompt}")
    print("\n🧠 SAFREE 토큰 복원 결과:")
    for i, mod_vec in enumerate(embeds_norm):
        if i >= rm_vector.shape[0]:
            break
        original_token = tokenizer.convert_ids_to_tokens([token_ids[i]])[0] if i < token_ids.shape[0] else "[PAD]"
        if rm_vector[i] == 1:  # 유지됨
            print(f"[{i}] 유지됨 ✅ → {original_token}")
        else:
            sims = torch.matmul(mod_vec, token_embeddings.T)
            top_id = torch.argmax(sims).item()
            token = id_to_token.get(top_id, "[UNK]")
            print(f"[{i}] 수정됨 ❌ : {original_token} → {token}")

def visualize_embeddings(original_embeds, modified_embeds, concept_matrix, rm_vector, max_tokens=20):
    original = original_embeds.squeeze(0).detach().cpu()
    modified = modified_embeds.squeeze(0).detach().cpu()
    concept_matrix = concept_matrix.detach().cpu()

    num_tokens = min(max_tokens, original.shape[0], modified.shape[0], rm_vector.shape[0])

    all_embeds = torch.cat([original[:num_tokens], modified[:num_tokens]], dim=0)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_embeds.numpy())

    orig_pca = pca_result[:num_tokens]
    mod_pca = pca_result[num_tokens:]

    plt.figure(figsize=(12, 6))

    for i in range(num_tokens):
        if rm_vector[i] == 1:
            # 유지된 토큰 (blue o)
            plt.scatter(orig_pca[i, 0], orig_pca[i, 1], color="blue", marker="o", label="Unchanged" if i == 0 else "", alpha=0.7)
        else:
            # 수정된 토큰 (원본: red o, 수정: orange x)
            plt.scatter(orig_pca[i, 0], orig_pca[i, 1], color="red", marker="o", label="Modified (Before)" if i == 0 else "", alpha=0.8)
            plt.scatter(mod_pca[i, 0], mod_pca[i, 1], color="orange", marker="x", label="Modified (After)" if i == 0 else "", alpha=0.9)
            plt.plot([orig_pca[i, 0], mod_pca[i, 0]], [orig_pca[i, 1], mod_pca[i, 1]], linestyle="--", color="gray", alpha=0.4)

    # 유해 서브스페이스 방향 벡터 시각화
    harmful_vec = concept_matrix @ torch.randn(concept_matrix.shape[0])
    harmful_vec = harmful_vec / harmful_vec.norm()
    harmful_2d = pca.transform(harmful_vec.unsqueeze(0).numpy())
    plt.scatter(harmful_2d[0, 0], harmful_2d[0, 1], color="black", marker="*", s=200, label="Harmful Subspace")

    plt.title("SAFREE Embedding Visualization (PCA 2D)")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("embedding_projection.png")
    print("📌 Embedding projection plot saved: embedding_projection.png")

def log_embeddings_tensorboard(original_embeds, modified_embeds, neg_embeds, rm_vector, tokenizer, prompt, negative_prompt_space):
    writer = SummaryWriter(log_dir="./runs/safree_tsne")

    orig = original_embeds.squeeze(0).detach().cpu()
    mod = modified_embeds.squeeze(0).detach().cpu()
    neg = neg_embeds.detach().cpu()

    tokenized = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt", add_special_tokens=False)
    token_ids = tokenized.input_ids.squeeze(0)

    all_embeds = []
    all_labels = []

    for i in range(orig.shape[0]):
        token_str = tokenizer.convert_ids_to_tokens(token_ids[i].item()) if i < token_ids.shape[0] else "[PAD]"
        
        if token_str in ["<|endoftext|>", "<|startoftext|>"]:
            continue
        
        if i < rm_vector.shape[0] and rm_vector[i] == 0:
            all_embeds.append(orig[i])
            all_labels.append(f"[M] {token_str}")
            all_embeds.append(mod[i])
            all_labels.append(f"[M'] {token_str}")
        else:
            all_embeds.append(orig[i])
            all_labels.append(f"[K] {token_str}")

    # 🔥 Harmful (negative prompt space)
    for vec, label in zip(neg, negative_prompt_space):
        all_embeds.append(vec)
        all_labels.append(f"[NEG] {label}")

    all_embeds = torch.stack(all_embeds)
    writer.add_embedding(all_embeds, metadata=all_labels, global_step=0, tag="safree")
    writer.close()
    print("✅ TensorBoard embedding logged with readable labels!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--category", type=str, default="nudity", choices=["nudity", "artists-VanGogh", "artists-KellyMcKernan"])
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    original_emb, modified_emb, rm_vector, project_matrix, neg_embeds, negative_prompt_space = get_safree_modified_embeddings(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=args.prompt,
        category=args.category,
        alpha=args.alpha,
        device=device
    )

    plot_cosine_similarity(original_emb, modified_emb)
    decode_nearest_tokens(modified_emb, original_emb, tokenizer, text_encoder, rm_vector, prompt=args.prompt)
    visualize_embeddings(original_emb, modified_emb, project_matrix, rm_vector)

    log_embeddings_tensorboard(original_emb, modified_emb, neg_embeds, rm_vector, tokenizer, prompt=args.prompt, negative_prompt_space=negative_prompt_space)

if __name__ == "__main__":
    main()