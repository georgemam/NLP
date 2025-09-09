from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import re

orig_text_1 = (
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. "
    "Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to the doctor, as his next contract checking, to all of us. "
    "I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. "
    "I am very appreciated the full support of the professor, for our Springer proceedings publication."
)

orig_text_2 = (
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing "
    "as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, "
    "they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. "
    "Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again. "
    "Because I didn't see that part final yet, or maybe I missed, I apologize if so. "
    "Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets."
)

sent_1A_1 = "Hope you too, to enjoy it as my deepest wishes."
sent_1A_2 = "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."

recon_1A_1 = "I hope you enjoy it too.My best wishes."
recon_1A_2 = "Anyway, I believe the team, although a bit of delay and less communication in recent days, they really tried their best on the paper and in our cooperation."


paraphrase_dict = {
    "text_1": {
        "original": orig_text_1,
        "Vamsi/T5_Paraphrase_Paws": "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. I got this message to see the approved message , in fact I received the message from the professor a couple of days ago to show me this . I am very appreciated the full support of the professor for our Springer proceedings publication .",
        "ramsrigouthamg/t5_paraphraser": "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. I am very appreciated the full support of the professor, for our Springer proceedings publication.",
        "prithivida/parrot_paraphraser_on_T5": "Today is our dragon boat festival in our Chinese culture to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. I got this message to see the approved message. In fact I received the message from the professor to show me this, a couple of days ago."
    },
    "text_2": {
        "original": orig_text_2,
        "Vamsi/T5_Paraphrase_Paws": "During our final discussion, I told him about the new submission — the one we were waiting for since last autumn , but the updates was confusing as it did not include the full feedback from reviewer or maybe editor ? Anyway, I think the team really tried best for paper and cooperation . We should be grateful, I mean all of us , for the acceptance and efforts until the Springer link finally came last week , I think . Also, kindly remind me if the doctor still plan for the acknowledgments section edit before",
        "ramsrigouthamg/t5_paraphraser": "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the doctor still plan for",
        "prithivida/parrot_paraphraser_on_T5": "I told him about the new submission — the one we were waiting since last autumn but the updates was confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link finally came last week. Also, please remind me if the doctor still plan for"
    }
}

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

analysis_texts = {
    "deliverable_1A_sentence_1": {
        "original": sent_1A_1,
        "custom_reconstruction": recon_1A_1
    },
    "deliverable_1A_sentence_2": {
        "original": sent_1A_2,
        "custom_reconstruction": recon_1A_2
    }
}
analysis_texts.update(paraphrase_dict)

sim_scores = {}

for group, items in analysis_texts.items():
    orig = items["original"]
    orig_emb = sbert_model.encode(orig, convert_to_tensor=True)
    sim_scores[group] = {}
    for key in items:
        if key == "original":
            continue
        emb = sbert_model.encode(items[key], convert_to_tensor=True)
        score = util.pytorch_cos_sim(orig_emb, emb).item()
        sim_scores[group][key] = score


print("\nAnalysis for Deliverable 1A")
print("-" * 50)
for k in ["deliverable_1A_sentence_1", "deliverable_1A_sentence_2"]:
    vals = sim_scores.get(k, {})
    if vals:
        print(f"\n{k.upper()} Original vs Custom Reconstruction: ")
        cs = vals.get("custom_reconstruction", "N/A")
        print(f"Cosine Similarity = {cs:.4f}")


print("\nAnalysis for Deliverable 1B")
print("-" * 50)
for k in ["text_1", "text_2"]:
    vals = sim_scores.get(k, {})
    if vals:
        print(f"\n{k.upper()} vs Pre-trained Models:")
        for idx, (mod, val) in enumerate(sorted(vals.items(), key=lambda x: x[1], reverse=True), 1):
            print(f"    {idx}. {mod}: {val:.4f}")


print("\nVisualizing Sentence Embeddings (PCA)...")

viz_texts, viz_labels, viz_colors, viz_markers = [], [], [], []

color_map = {
    "text_1": "blue",
    "text_2": "red",
    "deliverable_1A_sentence_1": "green",
    "deliverable_1A_sentence_2": "purple"
}
marker_map = {
    "original": "o",
    "Vamsi/T5_Paraphrase_Paws": "s",
    "ramsrigouthamg/t5_paraphraser": "^",
    "prithivida/paraphraser_on_T5": "D",
    "custom_reconstruction": "X"
}

for group, items in analysis_texts.items():
    col = color_map[group]
    for key, txt in items.items():
        viz_texts.append(txt)
        lbl = key
        if "deliverable_1A" in group:
            lbl = "Original" if key == "original" else "Custom Method"
        elif key == "original":
            lbl = "Original"
        viz_labels.append(f"{group.replace('deliverable_1A_sentence_', '1A Sent ').upper()} - {lbl}")
        viz_colors.append(col)
        viz_markers.append(marker_map.get(key, "o"))

embeddings = sbert_model.encode(viz_texts, convert_to_tensor=False)
pca_model = PCA(n_components=2, random_state=42)
pca_embeds = pca_model.fit_transform(embeddings)

plt.figure(figsize=(14, 10))
legend_dict = {}
for idx, lbl in enumerate(viz_labels):
    tup = (viz_colors[idx], viz_markers[idx])
    if tup not in legend_dict:
        legend_dict[tup] = lbl
    plt.scatter(pca_embeds[idx, 0], pca_embeds[idx, 1],
                color=viz_colors[idx], marker=viz_markers[idx], s=100)

plt.title('PCA of Sentence Embeddings (All Reconstructions)', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.grid(True)

legend_items = []
for (col, mark), lbl in legend_dict.items():
    legend_items.append(plt.Line2D([0], [0], marker=mark, color='w', label=lbl,
                                   markerfacecolor=col, markersize=10))

plt.legend(handles=legend_items, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
plt.tight_layout()
plt.show()
