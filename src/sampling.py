"""
Hybrid Undersampling and Oversampling Module.
Implements:
- Algorithm 1: Tomek Links Undersampling Algorithm (Section V.B)
- Algorithm 2: Borderline-SMOTE Algorithm (Section V.C.2)
- Algorithm 3: BIRCH Clustering Borderline SMOTE (BCBSMOTE) Algorithm (Section V.C.3)
- HybridTomekBCBSMOTE: Combined proposed pipeline (Figure 5)
- Comparative Baseline Samplers for benchmarking (Table 4)
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import Birch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class TomekLinksUndersampler:
    """
    Algorithm 1: Tomek Links Undersampling Algorithm
    Paper Section V.B (Page 7)
    
    Identifies pairs (x, y) where:
    - x belongs to the majority class (Class 0, legitimate)
    - y belongs to the minority class (Class 1, fraud)
    - x and y are mutual nearest neighbors.
    
    Removes x (the majority instance) to eliminate borderline noise.
    """
    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric
        self.removed_indices_ = []

    def fit_resample(self, X, y):
        is_df = isinstance(X, pd.DataFrame)
        cols = X.columns if is_df else None
        
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int32)
        
        maj_mask = (y_arr == 0)
        min_mask = (y_arr == 1)
        
        X_maj = X_arr[maj_mask]
        X_min = X_arr[min_mask]
        maj_indices = np.where(maj_mask)[0]
        
        if len(X_maj) == 0 or len(X_min) == 0:
            return X, y
            
        print(f"[TomekLinks] Starting undersampling on {len(X_maj):,} majority and {len(X_min):,} minority instances...")
        
        # Standardize features for accurate Euclidean nearest-neighbor distances
        scaler = StandardScaler()
        X_maj_s = scaler.fit_transform(X_maj)
        X_min_s = scaler.transform(X_min)
        
        # Fit NN on majority and minority
        nn_maj = NearestNeighbors(n_neighbors=min(2, len(X_maj)), metric=self.metric, n_jobs=-1).fit(X_maj_s)
        nn_min = NearestNeighbors(n_neighbors=min(2, len(X_min)), metric=self.metric, n_jobs=-1).fit(X_min_s)
        
        # 1. For each minority sample, find distance to nearest fellow minority sample
        if len(X_min) > 1:
            d_min_self, _ = nn_min.kneighbors(X_min_s, n_neighbors=2)
            dist_to_nearest_min = d_min_self[:, 1]
        else:
            dist_to_nearest_min = np.full(len(X_min), np.inf)
            
        # 2. For each minority sample, find nearest majority sample
        dist_to_maj, idx_maj_nn = nn_maj.kneighbors(X_min_s, n_neighbors=1)
        dist_to_maj = dist_to_maj[:, 0]
        idx_maj_nn = idx_maj_nn[:, 0]
        
        # Candidates: minority points whose 1-NN in the whole dataset is a majority point
        cand_mask = dist_to_maj < dist_to_nearest_min
        cand_y_indices = np.where(cand_mask)[0]
        cand_maj_indices = idx_maj_nn[cand_y_indices]
        
        unique_cand_maj = np.unique(cand_maj_indices)
        
        # 3. For each candidate majority point, verify if its 1-NN in the entire dataset is that minority point
        if len(unique_cand_maj) > 0:
            d_x_maj, _ = nn_maj.kneighbors(X_maj_s[unique_cand_maj], n_neighbors=min(2, len(X_maj)))
            dist_x_to_maj = d_x_maj[:, 1] if len(X_maj) > 1 else np.full(len(unique_cand_maj), np.inf)
            
            d_x_min, idx_x_min = nn_min.kneighbors(X_maj_s[unique_cand_maj], n_neighbors=1)
            dist_x_to_min = d_x_min[:, 0]
            nearest_min_to_x = idx_x_min[:, 0]
            
            tomek_maj_local = []
            for i, maj_idx in enumerate(unique_cand_maj):
                # Is minority point closer to x than x's nearest majority neighbor?
                if dist_x_to_min[i] < dist_x_to_maj[i]:
                    min_idx = nearest_min_to_x[i]
                    # Check mutual condition
                    if cand_mask[min_idx] and idx_maj_nn[min_idx] == maj_idx:
                        tomek_maj_local.append(maj_idx)
                        
            self.removed_indices_ = maj_indices[tomek_maj_local]
        else:
            self.removed_indices_ = np.array([], dtype=int)
            
        n_removed = len(self.removed_indices_)
        print(f"[TomekLinks] Removed {n_removed:,} noise-generating majority instances from Tomek links.")
        print(f"[TomekLinks] Remaining majority: {len(X_maj) - n_removed:,}, Minority: {len(X_min):,}")
        
        keep_mask = np.ones(len(X_arr), dtype=bool)
        keep_mask[self.removed_indices_] = False
        
        X_res = X_arr[keep_mask]
        y_res = y_arr[keep_mask]
        
        if is_df:
            return pd.DataFrame(X_res, columns=cols), pd.Series(y_res, name='isFraud')
        return X_res, y_res


class BCBSMOTEOversampler:
    """
    Algorithms 2 & 3: BIRCH Clustering Borderline SMOTE (BCBSMOTE) Algorithm
    Paper Section V.C (Pages 7-9)
    
    Steps:
    1. Employ BIRCH clustering on X_train with y_train=1 (minority class).
    2. Identify minority class clusters with size >= cluster_size (default: 5).
    3. Based on sample percentage in each cluster, allocate target synthetic count.
    4. Apply Borderline-SMOTE locally to each cluster to generate synthetic samples.
    5. Combine synthetic samples with original dataset.
    """
    def __init__(self, cluster_size: int = 5, m_neighbors: int = 5, k_neighbors: int = 5,
                 threshold: float = 0.5, branching_factor: int = 50,
                 sampling_ratio: float = 0.02, random_state: int = 42):
        """
        Parameters:
        -----------
        cluster_size : int
            Minimum cluster size for BIRCH clustering (clusters < 5 are dropped as noise).
        m_neighbors : int
            Number of neighbors to evaluate Danger / Safe / Noise in Borderline-SMOTE (Algorithm 2).
        k_neighbors : int
            Number of nearest neighbors inside the cluster used for interpolation.
        threshold : float
            BIRCH clustering threshold (radius of the subcluster).
        branching_factor : int
            Maximum number of CF subclusters in each node.
        sampling_ratio : float
            Target ratio of minority to majority samples (e.g. 0.02, 0.1, or 1.0 for balanced).
        random_state : int
            Reproducibility seed.
        """
        self.cluster_size = cluster_size
        self.m_neighbors = m_neighbors
        self.k_neighbors = k_neighbors
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.sampling_ratio = sampling_ratio
        self.random_state = random_state

    def fit_resample(self, X, y):
        is_df = isinstance(X, pd.DataFrame)
        cols = X.columns if is_df else None
        
        rng = np.random.RandomState(self.random_state)
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int32)
        
        maj_mask = (y_arr == 0)
        min_mask = (y_arr == 1)
        
        X_maj = X_arr[maj_mask]
        X_min = X_arr[min_mask]
        
        n_maj = len(X_maj)
        n_min = len(X_min)
        
        # Determine number of synthetic samples to generate
        target_min_count = int(n_maj * self.sampling_ratio)
        n_to_generate = max(0, target_min_count - n_min)
        
        print(f"[BCBSMOTE] Original counts -> Majority: {n_maj:,}, Minority: {n_min:,}")
        print(f"[BCBSMOTE] Target minority count (ratio {self.sampling_ratio}): {target_min_count:,}")
        print(f"[BCBSMOTE] Synthetic samples to generate: {n_to_generate:,}")
        
        if n_to_generate == 0 or n_min < self.cluster_size:
            return X, y
            
        # Step 1: BIRCH clustering on minority instances (y_train = 1)
        scaler = StandardScaler()
        X_min_scaled = scaler.fit_transform(X_min)
        
        birch = Birch(
            threshold=self.threshold,
            branching_factor=self.branching_factor,
            n_clusters=None
        )
        cluster_labels = birch.fit_predict(X_min_scaled)
        
        # Step 2: Identify minority clusters with size >= cluster_size
        counts = Counter(cluster_labels)
        valid_clusters = [c for c, count in counts.items() if count >= self.cluster_size]
        filtered_clusters = [c for c, count in counts.items() if count < self.cluster_size]
        
        print(f"[BCBSMOTE] Total BIRCH clusters formed: {len(counts)}")
        print(f"[BCBSMOTE] Valid clusters (size >= {self.cluster_size}): {len(valid_clusters)}")
        print(f"[BCBSMOTE] Noise clusters filtered out (size < {self.cluster_size}): {len(filtered_clusters)}")
        
        if not valid_clusters:
            valid_clusters = list(counts.keys())
            
        total_valid_samples = sum(counts[c] for c in valid_clusters)
        
        # Algorithm 2: Borderline-SMOTE labeling (Danger, Safe, Noise)
        # Query closest m samples from the available dataset
        maj_sub_size = min(len(X_maj), 250000)
        maj_sub_idx = rng.choice(len(X_maj), size=maj_sub_size, replace=False)
        X_maj_ref = scaler.transform(X_maj[maj_sub_idx])
        
        m = self.m_neighbors
        nn_maj = NearestNeighbors(n_neighbors=m, n_jobs=-1).fit(X_maj_ref)
        nn_min_all = NearestNeighbors(n_neighbors=m+1, n_jobs=-1).fit(X_min_scaled)
        
        dist_maj, _ = nn_maj.kneighbors(X_min_scaled, n_neighbors=m)
        dist_min, _ = nn_min_all.kneighbors(X_min_scaled, n_neighbors=m+1)
        
        # Label minority samples
        is_danger = np.zeros(n_min, dtype=bool)
        for i in range(n_min):
            all_dists = [(dist_maj[i, j], 0) for j in range(m)] + [(dist_min[i, j+1], 1) for j in range(m)]
            all_dists.sort(key=lambda t: t[0])
            top_m_classes = [c for _, c in all_dists[:m]]
            m_prime = top_m_classes.count(0)
            
            # If m/2 <= m' < m: Danger (border sample)
            if (m / 2.0) <= m_prime < m:
                is_danger[i] = True
                
        print(f"[BCBSMOTE] Danger (borderline) minority samples identified: {np.sum(is_danger):,} / {n_min:,}")
        
        # Steps 3, 4, 5: Generate samples per cluster
        synthetic_samples = []
        for c in valid_clusters:
            c_indices = np.where(cluster_labels == c)[0]
            cluster_share = len(c_indices) / total_valid_samples
            n_cluster_gen = int(round(n_to_generate * cluster_share))
            if n_cluster_gen <= 0:
                continue
                
            X_cluster = X_min[c_indices]
            danger_in_cluster = c_indices[is_danger[c_indices]]
            
            # Seed selection: prioritize Danger samples if present in cluster
            if len(danger_in_cluster) > 0:
                seed_indices = danger_in_cluster
            else:
                seed_indices = c_indices
                
            k = min(self.k_neighbors, len(X_cluster) - 1)
            if k < 1:
                # Add tiny jitter if cluster has single sample
                std_noise = np.std(X_min, axis=0, keepdims=True)
                noise = rng.normal(0, 0.005, size=(n_cluster_gen, X_arr.shape[1])) * std_noise
                synthetic_samples.append(X_min[c_indices[0]] + noise)
                continue
                
            nn_cluster = NearestNeighbors(n_neighbors=k+1).fit(X_cluster)
            _, neighbors_idx = nn_cluster.kneighbors(X_min[seed_indices], n_neighbors=k+1)
            
            chosen_seed_idxs = rng.choice(len(seed_indices), size=n_cluster_gen, replace=True)
            chosen_neighbor_order = rng.randint(1, k+1, size=n_cluster_gen)
            betas = rng.uniform(0, 1, size=(n_cluster_gen, 1))
            
            base_pts = X_min[seed_indices[chosen_seed_idxs]]
            neighbor_pts = X_cluster[neighbors_idx[chosen_seed_idxs, chosen_neighbor_order]]
            
            cluster_synthetic = base_pts + betas * (neighbor_pts - base_pts)
            synthetic_samples.append(cluster_synthetic)
            
        if synthetic_samples:
            X_synth = np.vstack(synthetic_samples)
            y_synth = np.ones(len(X_synth), dtype=np.int32)
            X_res = np.vstack([X_arr, X_synth])
            y_res = np.concatenate([y_arr, y_synth])
            print(f"[BCBSMOTE] Successfully generated {len(X_synth):,} synthetic minority samples.")
        else:
            X_res, y_res = X_arr, y_arr
            
        print(f"[BCBSMOTE] Final resampled dataset: {len(X_res):,} samples (Normal: {np.sum(y_res==0):,}, Fraud: {np.sum(y_res==1):,})")
        
        if is_df:
            return pd.DataFrame(X_res, columns=cols), pd.Series(y_res, name='isFraud')
        return X_res, y_res


class HybridTomekBCBSMOTE:
    """
    Proposed Hybrid Resampling Architecture (Figure 5 in paper):
    1. Undersample majority instances using Tomek Links to remove boundary noise.
    2. Oversample minority instances using BIRCH Clustering Borderline-SMOTE (BCBSMOTE).
    """
    def __init__(self, cluster_size: int = 5, m_neighbors: int = 5, k_neighbors: int = 5,
                 threshold: float = 0.5, branching_factor: int = 50,
                 sampling_ratio: float = 0.02, random_state: int = 42):
        self.undersampler = TomekLinksUndersampler()
        self.oversampler = BCBSMOTEOversampler(
            cluster_size=cluster_size,
            m_neighbors=m_neighbors,
            k_neighbors=k_neighbors,
            threshold=threshold,
            branching_factor=branching_factor,
            sampling_ratio=sampling_ratio,
            random_state=random_state
        )

    def fit_resample(self, X, y):
        print("=== Step 1: Applying Tomek Links Majority Undersampling ===")
        X_tl, y_tl = self.undersampler.fit_resample(X, y)
        print("\n=== Step 2: Applying BIRCH Clustering Borderline SMOTE (BCBSMOTE) ===")
        X_res, y_res = self.oversampler.fit_resample(X_tl, y_tl)
        return X_res, y_res


def get_baseline_sampler(sampler_name: str, sampling_ratio: float = 0.02, random_state: int = 42):
    """
    Factory helper to instantiate baseline samplers for Table 4 comparison:
    - SMOTE
    - B-SMOTE (BorderlineSMOTE)
    - ADASYN
    - SMOTE-Tomek (SMOTETomek)
    - SMOTEENN
    - Proposed (HybridTomekBCBSMOTE)
    """
    name = sampler_name.upper().replace('_', '-').replace(' ', '-')
    
    if name in ['PROPOSED', 'HYBRID-TOMEK-BCBSMOTE', 'BCBSMOTE']:
        return HybridTomekBCBSMOTE(sampling_ratio=sampling_ratio, random_state=random_state)
        
    try:
        from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
        from imblearn.combine import SMOTETomek, SMOTEENN
    except ImportError:
        raise ImportError("Please install imbalanced-learn to run baseline comparisons: pip install imbalanced-learn")
        
    if name == 'SMOTE':
        return SMOTE(sampling_strategy=sampling_ratio, random_state=random_state)
    elif name in ['B-SMOTE', 'BORDERLINE-SMOTE', 'BORDERLINESMOTE']:
        return BorderlineSMOTE(sampling_strategy=sampling_ratio, random_state=random_state)
    elif name == 'ADASYN':
        return ADASYN(sampling_strategy=sampling_ratio, random_state=random_state)
    elif name in ['SMOTE-TOMEK', 'SMOTETOMEK']:
        return SMOTETomek(sampling_strategy=sampling_ratio, random_state=random_state)
    elif name in ['SMOTEENN', 'SMOTE-ENN']:
        return SMOTEENN(sampling_strategy=sampling_ratio, random_state=random_state)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}. Choose from: SMOTE, B-SMOTE, ADASYN, SMOTE-Tomek, SMOTEENN, Proposed")
