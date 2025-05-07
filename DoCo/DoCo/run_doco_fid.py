import numpy as np
from scipy.linalg import sqrtm

def calculate_fid(feats1, feats2):
    mu1, mu2 = feats1.mean(axis=0), feats2.mean(axis=0)
    sigma1, sigma2 = np.cov(feats1, rowvar=False), np.cov(feats2, rowvar=False)
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# 🔽 기존 저장된 피처 불러오기
gen_feats = np.load("generated_features_list.npy", allow_pickle=True)
ref_feats = np.load("reference_features_list.npy", allow_pickle=True)

# 🔽 둘 중 짧은 길이로 잘라서 정렬
min_len = min(len(gen_feats), len(ref_feats))
gen_feats = np.array(gen_feats[:min_len])
ref_feats = np.array(ref_feats[:min_len])

# ✅ FID 계산
fid_score = calculate_fid(gen_feats, ref_feats)
print(f"✅ Calculated FID for {min_len} samples: {fid_score}")
