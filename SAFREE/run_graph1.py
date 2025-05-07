import pandas as pd
import matplotlib.pyplot as plt
import os

# 경로 설정
csv_path = "../scripts/result/SAFREE_SDv1-4_custom/result.csv"
output_dir = "../scripts/result/SAFREE_SDv1-4_custom"        
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# text_sim_to_nude 기준 정렬
df_sorted = df.sort_values(by="text_sim_to_nude", ascending=False)

# 시각화용 데이터
labels = df_sorted["word"].tolist()
sim_scores = df_sorted["text_sim_to_nude"].tolist()
nudenet_scores = df_sorted["avg_nudenet_score"].tolist()
clip_scores = df_sorted["avg_clip_score"].tolist()
unsafe_ratios = df_sorted["unsafe_ratio"].tolist()

# 시각화
fig, ax1 = plt.subplots(figsize=(20, 6))
ax1.bar(range(len(labels)), sim_scores, color='lightblue', label="Similarity (bar)")
ax1.set_xlabel("Expression")
ax1.set_ylabel("Embedding Similarity", color='blue')
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=90, fontsize=7)

# 두 번째 축 (NudeNet + CLIP)
ax2 = ax1.twinx()
ax2.plot(range(len(labels)), nudenet_scores, color='red', marker='o', label="NudeNet Score (avg)")
ax2.plot(range(len(labels)), clip_scores, color='green', marker='x', label="CLIP Score (avg)")
ax2.set_ylabel("Score", color='black')

# 범례
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title("Similarity vs NudeNet/CLIP Scores")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "semantic_all.png"))
plt.show()