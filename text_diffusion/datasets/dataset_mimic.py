import torch
import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
# from vocab import Vocab  # 暂时注释掉，因为MIMIC数据集不需要这个
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.medcode import InnerMap
from .patient_embedding_v2 import PatientEmbeddingV2


class MIMICDrugDataset(Dataset):
    """
    MIMIC-IV Drug Recommendation Dataset for Diffusion Model

    This dataset processes MIMIC-IV data to create:
    - Drug combination vectors (multi-hot encoding of ATC level 3 drugs)
    - Patient condition embeddings (diagnosis, procedures, drug history)
    - Compatible with text_diffusion framework
    """

    def __init__(self, root='./datasets', split='train', max_drugs=190, condition_dim=1024, 
                 mimic_root='/mnt/share/Zhiwen/mimic-iv-2.2/hosp'):
        assert split in {'train', 'valid', 'test'}
        self.root = os.path.join(root, 'mimic_drugs')
        self.split = split
        self.max_drugs = max_drugs
        self.condition_dim = condition_dim
        self.mimic_root = mimic_root

        # Create directory if not exists
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        # Load or create drug vocabulary
        self.drug_vocab = self._load_or_create_drug_vocab()

        # Load patients and admissions data for PatientEmbeddingV2
        print(f"Loading patients and admissions data from {mimic_root}...")
        patients_file = os.path.join(mimic_root, 'patients.csv.gz')
        admissions_file = os.path.join(mimic_root, 'admissions.csv.gz')
        
        self.patients_df = pd.read_csv(patients_file, compression='gzip')
        self.admissions_df = pd.read_csv(admissions_file, compression='gzip')
        print(f"✓ Loaded {len(self.patients_df)} patients and {len(self.admissions_df)} admissions")

        # Initialize patient embedding module (V2 with 1024-dim)
        self.patient_embedding = PatientEmbeddingV2(
            drug_vocab=self.drug_vocab,
            vocab_dir=self.root,  # 指向mimic_drugs目录
            patients_df=self.patients_df,
            admissions_df=self.admissions_df,
            condition_dim=condition_dim
        )

        # Load or create condition mappings
        self.diagnosis_map = InnerMap.load('ICD9CM')
        self.procedure_map = InnerMap.load('ICD9PROC')

        # Preprocess data if not exists
        if not os.path.exists(self.processed_file(split)):
            self._preprocess_data(split)

        # Load processed data
        self.data = torch.load(self.processed_file(split))
        self.conditions = torch.load(self.condition_file(split))
        self.sample_metadata = torch.load(self.metadata_file(split))  # subject_id, hadm_id

    def __getitem__(self, index):
        # Return drug vector as indices (0 or 1 for each drug position)
        drug_vector = self.data[index]  # Shape: (190,)
        # Convert multi-hot to indices: 0 = not prescribed, 1 = prescribed
        drug_indices = drug_vector.long()  # Convert to long tensor

        # Get condition embedding using the new embedding method
        condition_embedding = self.conditions[index]  # Shape: (condition_dim,)

        return drug_indices, condition_embedding, self.max_drugs

    def __len__(self):
        return len(self.data)

    def _load_or_create_drug_vocab(self):
        """Load or create ATC level 3 drug vocabulary"""
        vocab_file = os.path.join(self.root, 'drug_vocab.json')

        if os.path.exists(vocab_file):
            with open(vocab_file, 'r') as f:
                drug_vocab = json.load(f)
            return drug_vocab

        # Create drug vocabulary from MIMIC data
        print("Creating drug vocabulary...")
        drug_vocab = self._create_drug_vocab()

        # Save vocabulary
        with open(vocab_file, 'w') as f:
            json.dump(drug_vocab, f, indent=4)

        return drug_vocab

    def _create_drug_vocab(self):
        """Create drug vocabulary from MIMIC-IV data"""
        # Load MIMIC data
        mimic4_ds = MIMIC4Dataset(
            root=self.mimic_root,
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            dev=False,
        )

        # Apply drug recommendation task
        from pyhealth.tasks import drug_recommendation_mimic4_fn
        mimic4_ds_dr = mimic4_ds.set_task(task_fn=drug_recommendation_mimic4_fn)

        # Count drug frequencies
        drug_counts = {}
        for sample in mimic4_ds_dr.samples:
            for drug in sample['drugs']:
                drug_counts[drug] = drug_counts.get(drug, 0) + 1

        # Create vocabulary from all drugs (should be 189 ATC level 3 drugs based on build_vocabularies.py)
        drug_vocab = {drug: idx for idx, drug in enumerate(sorted(drug_counts.keys()))}

        print(f"Created drug vocabulary with {len(drug_vocab)} drugs")
        return drug_vocab

    def _preprocess_data(self, split):
        """Preprocess MIMIC data for the specified split"""
        print(f"Preprocessing {split} data...")

        # Load MIMIC data
        mimic4_ds = MIMIC4Dataset(
            root=self.mimic_root,
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            dev=False,
        )

        # Apply drug recommendation task
        from pyhealth.tasks import drug_recommendation_mimic4_fn
        mimic4_ds_dr = mimic4_ds.set_task(task_fn=drug_recommendation_mimic4_fn)

        # Split data
        total_samples = len(mimic4_ds_dr.samples)
        if split == 'train':
            samples = mimic4_ds_dr.samples[:int(0.7 * total_samples)]
        elif split == 'valid':
            samples = mimic4_ds_dr.samples[int(0.7 * total_samples):int(0.85 * total_samples)]
        else:  # test
            samples = mimic4_ds_dr.samples[int(0.85 * total_samples):]

        print(f"Processing {len(samples)} samples for {split} split")

        # Process samples
        drug_vectors = []
        condition_embeddings = []
        sample_metadata = []  # Store (subject_id, hadm_id) for each sample

        for i, sample in enumerate(samples):
            if i % 1000 == 0:
                print(f"Processing sample {i}/{len(samples)}")

            # Extract subject_id and hadm_id
            subject_id = sample.get('subject_id', sample.get('patient_id', None))
            hadm_id = sample.get('visit_id', None)  # PyHealth uses 'visit_id' for hadm_id
            
            if subject_id is None or hadm_id is None:
                print(f"Warning: Missing subject_id or hadm_id for sample {i}, skipping")
                continue

            # Create drug vector
            drug_vector = self._create_drug_vector(sample['drugs'])
            drug_vectors.append(drug_vector)

            # Create condition embedding with subject_id and hadm_id
            condition_embedding = self._create_condition_embedding(sample, subject_id, hadm_id)
            condition_embeddings.append(condition_embedding)
            
            # Store metadata
            sample_metadata.append((subject_id, hadm_id))

        # Convert to tensors
        drug_vectors = torch.stack(drug_vectors)
        condition_embeddings = torch.stack(condition_embeddings)

        # Save processed data
        torch.save(drug_vectors, self.processed_file(split))
        torch.save(condition_embeddings, self.condition_file(split))
        torch.save(sample_metadata, self.metadata_file(split))

        print(f"Saved {len(drug_vectors)} samples to {self.processed_file(split)}")

    def _create_drug_vector(self, drugs):
        """Create multi-hot drug vector"""
        vector = torch.zeros(self.max_drugs)
        for drug in drugs:
            if drug in self.drug_vocab:
                vector[self.drug_vocab[drug]] = 1
        return vector

    def _create_condition_embedding(self, sample, subject_id, hadm_id):
        """Create patient condition embedding using PatientEmbeddingV2"""
        # Use the new PatientEmbedding module with subject_id and hadm_id
        return self.patient_embedding.create_condition_embedding(sample, subject_id, hadm_id)

    def processed_file(self, split):
        return os.path.join(self.root, f'processed_{split}.pt')

    def condition_file(self, split):
        return os.path.join(self.root, f'conditions_{split}.pt')
    
    def metadata_file(self, split):
        return os.path.join(self.root, f'metadata_{split}.pt')
