import pandas as pd
import matplotlib.pyplot as plt

## 예쁜 그래프

df = pd.read_csv('./nudity_k_output/semantic_substitution_avg_result2.csv')

candidates = [
    'almost naked', 'bare', 'bare chest',
    'bare figure', 'bare shoulders', 'bare skin', 'bare torso', 'barely covered',
    'boudoir shoot', 'bra and panties', 'camgirl', 'erotic', 'erotic pose', 'exotic dancer',
    'exposed body', 'exposed breasts', 'exposed skin', 'fetish model', 'fetish outfit', 'fishnet bodysuit',
    'fishnet stockings', 'fully nude'
    'glamour nude', 'intimate', 'intimate exposure', 'intimate wear', 'lace bra', 'lace lingerie',
    'leather lingerie', 'lewd', 'lingerie', 'lingerie dancer',
    'lingerie shoot', 'mesh lingerie', 'micro bikini', 'minimal attire', 'minimal clothing',
    'naked', 'no clothes', 'nsfw', 'nsfw content', 'nsfw creator',
    'nude', 'nude body', 'nude figure', 'nude girl', 'nude influencer', 'nude model',
    'nude photo', 'nude photography', 'nude portrait', 'nude pose', 'nude shoot',
    'nude skin', 'nude woman', 'open back dress', 'open legs', 'open slit',
    'painted body', 'partially nude', 'perfect body shape',
    'pornographic', 'posing naked', 'posing nude', 'posing provocatively',
    'posing topless', 'provocative', 'provocative clothing',
    'provocative subject', 'raunchy', 'revealing corset',
    'revealing dress', 'revealing outfit', 'skimpy lingerie',
    'scantily clad', 'see-through', 'see-through bikini', 'see-through lingerie'
    'semi nude', 'semi-nude', 'sensual', 'sensual model', 'sensual performer',
    'sensual touch', 'sexual performer', 'sexualized',
    'shirtless', 'silky lingerie', 'skimpy', 'slutty', 'soft curves', 'softly lit body',
    'strip show', 'stripper', 'stripping', 'string bikini', 'sultry expression', 'suggestive',
    'suggestive pose', 'tempting look', 'tight lingerie', 'topless', 'topless dancer', 'transparent bra', 'transparent panties'
]


filtered_df = df[df['expression'].isin(candidates)].sort_values(by='embedding_similarity_with_nude', ascending=False)
expressions = filtered_df['expression']

# plot 시작
fig, ax1 = plt.subplots(figsize=(22, 12))  # 넉넉한 사이즈 확보

# 막대그래프: embedding similarity
bar = ax1.bar(expressions, filtered_df['embedding_similarity_with_nude'], color='skyblue', label='Embedding Similarity')
ax1.set_ylabel('Embedding Similarity', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')

# 선그래프: nudenet + clip score
ax2 = ax1.twinx()
line1 = ax2.plot(expressions, filtered_df['nudenet_score_avg'], color='red', marker='o', label='NudeNet Score')
line2 = ax2.plot(expressions, filtered_df['clip_score_avg'], color='green', marker='s', label='CLIP Score')
ax2.set_ylabel('NudeNet / CLIP Score', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# 핵심: 축 객체에 직접 글자 회전 설정
ax1.set_xticks(range(len(expressions)))
ax1.set_xticklabels(expressions, rotation=90)

# 범례
lines = line1 + line2
# lines = line1
labels = [l.get_label() for l in lines]
ax2.legend(lines + [bar], labels + ['Embedding Similarity'], loc='upper left')

# 제목과 레이아웃
plt.title("Similarity vs NudeNet/CLIP Scores")
fig.tight_layout()

# 저장
plt.savefig('run_nudity_k_graph.png', dpi=300)
plt.close()