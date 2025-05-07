import nltk
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn

# WordNet
nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv("../nudity_k_output/semantic_substitution_avg_result4.csv")

expressions = df[df['expression'] != "nude"].sort_values(
    by="nudenet_score_avg", ascending=False
).head(20)['expression'].tolist()

central_word = "nude"

# WordNet synset 경로 계산
def get_common_hypernym_path(w1, w2):
    syns1 = wn.synsets(w1, pos=wn.NOUN)
    syns2 = wn.synsets(w2, pos=wn.NOUN)
    for s1 in syns1:
        for s2 in syns2:
            lcs = s1.lowest_common_hypernyms(s2)
            if lcs:
                return [w1, lcs[0].name().split('.')[0], w2]
    return None

# 그래프 생성
G = nx.DiGraph()
G.add_node(central_word)

for word in expressions:
    path = get_common_hypernym_path(central_word, word)
    if path:
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i + 1])

plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, k=0.6)
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=1000, font_size=10)
plt.title("WordNet Semantic Tree Centered on 'nude'")
plt.savefig("nude_semantic_tree.png", dpi=300)
plt.close()