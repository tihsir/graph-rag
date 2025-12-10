# Long-Form RAG Evaluation Results

**Date:** December 2024  
**Questions Evaluated:** 20  
**Judge Model:** Gemini 2.0 Flash

---

## 🏆 Final Results Summary

| Rank | System | Average Score |
|------|--------|---------------|
| 🥇 | **SARG+ORAG** | **9.86/10** |
| 🥈 | Vanilla RAG | 9.68/10 |
| 🥉 | KG-RAG | 9.66/10 |

---

## 📊 Per-Question Scores

| # | Question ID | Topic | Vanilla | KG-RAG | SARG+ORAG |
|---|-------------|-------|---------|--------|-----------|
| 1 | q1_hla_psoriasis | HLA-B & Psoriasis | 9.8 | 10.0 | 9.5 |
| 2 | q2_diabetes_nod2 | Diabetes & NOD2 | 8.8 | 9.5 | 10.0 |
| 3 | q3_multi_disease | Multi-disease comparison | 8.2 | 8.8 | 9.2 |
| 4 | q4_brca_cancer | BRCA & Cancer | 10.0 | 10.0 | 10.0 |
| 5 | q5_alzheimer_apoe | APOE & Alzheimer's | 10.0 | 10.0 | 10.0 |
| 6 | q6_cystic_fibrosis | Cystic Fibrosis | 10.0 | 10.0 | 10.0 |
| 7 | q7_parkinsons_genetics | Parkinson's Genetics | 10.0 | 10.0 | 10.0 |
| 8 | q8_rheumatoid_arthritis | Rheumatoid Arthritis | 10.0 | 10.0 | 10.0 |
| 9 | q9_sickle_cell | Sickle Cell Disease | 10.0 | 10.0 | 10.0 |
| 10 | q10_schizophrenia | Schizophrenia Genetics | 9.8 | 9.5 | 9.8 |
| 11 | q11_inflammatory_bowel | IBD Comparison | 9.2 | 9.2 | 9.5 |
| 12 | q12_hypertension | Hypertension Genetics | 9.5 | 9.0 | 9.5 |
| 13 | q13_lupus | SLE & Complement | 9.5 | 9.2 | 10.0 |
| 14 | q14_hemophilia | Hemophilia A vs B | 10.0 | 10.0 | 10.0 |
| 15 | q15_asthma_genetics | Asthma Genetics | 9.8 | 9.5 | 10.0 |
| 16 | q16_colorectal_cancer | CRC & Lynch Syndrome | 9.8 | 9.8 | 10.0 |
| 17 | q17_muscular_dystrophy | DMD & Dystrophin | 10.0 | 10.0 | 10.0 |
| 18 | q18_thyroid_disease | Graves' vs Hashimoto's | 9.5 | 9.5 | 10.0 |
| 19 | q19_epilepsy | Epilepsy & Ion Channels | 10.0 | 8.8 | 9.5 |
| 20 | q20_obesity_genetics | Obesity & FTO/MC4R/LEP | 9.8 | 9.8 | 10.0 |

---

## 🔬 Triple Extraction Statistics

### Total Triples Extracted Across All Questions

| Metric | Count |
|--------|-------|
| Total KG Triples | 1,339 |
| Total Semantic Triples | 1,264 |
| Total Duplicates Found | 57 |
| Total Merged Triples | 2,546 |

### Per-Question Triple Analysis

| Question | KG Triples | Sem Triples | Duplicates | Merged |
|----------|------------|-------------|------------|--------|
| q1_hla_psoriasis | 76 | 67 | 0 | 143 |
| q2_diabetes_nod2 | 63 | 61 | 0 | 124 |
| q3_multi_disease | 69 | 83 | 7 | 145 |
| q4_brca_cancer | 65 | 58 | 3 | 120 |
| q5_alzheimer_apoe | 73 | 67 | 4 | 136 |
| q6_cystic_fibrosis | 66 | 52 | 11 | 110 |
| q7_parkinsons_genetics | 69 | 69 | 0 | 138 |
| q8_rheumatoid_arthritis | 68 | 65 | 4 | 129 |
| q9_sickle_cell | 49 | 62 | 2 | 110 |
| q10_schizophrenia | 71 | 68 | 0 | 139 |
| q11_inflammatory_bowel | 62 | 59 | 5 | 116 |
| q12_hypertension | 67 | 63 | 2 | 128 |
| q13_lupus | 58 | 55 | 3 | 110 |
| q14_hemophilia | 45 | 48 | 4 | 89 |
| q15_asthma_genetics | 72 | 69 | 5 | 136 |
| q16_colorectal_cancer | 74 | 51 | 0 | 125 |
| q17_muscular_dystrophy | 57 | 40 | 6 | 92 |
| q18_thyroid_disease | 59 | 63 | 2 | 120 |
| q19_epilepsy | 85 | 80 | 0 | 165 |
| q20_obesity_genetics | 81 | 86 | 1 | 166 |

---

## 📋 Sample Triple Outputs

### Question 1: HLA-B & Psoriasis

**KG Triples (Sample):**
```
• (autoimmune disease) --[ASSOCIATES_WITH]--> (HLA-B)
• (autoimmune disease) --[ASSOCIATES_WITH]--> (HLA-A)
• (autoimmune disease) --[ASSOCIATES_WITH]--> (HLA-C)
• (autoimmune disease) --[ASSOCIATES_WITH]--> (HLA-DPB1)
• (autoimmune disease) --[ASSOCIATES_WITH]--> (HLA-DQB1)
• (autoimmune disease) --[ASSOCIATES_WITH]--> (IL1B)
• (autoimmune disease) --[ASSOCIATES_WITH]--> (TNF)
```

**Semantic Triples (Sample):**
```
• (rs13202464 x rs9267677) --[ASSOCIATES_WITH]--> (psoriasis)
• (rs17728338 x rs10484554) --[ASSOCIATES_WITH]--> (psoriasis)
• (rs6911408 x rs9468932) --[ASSOCIATES_WITH]--> (psoriasis)
```

**LLM Deduplication Reasoning:**
> "After careful analysis, no triples from the KG set express the same relationship with the same entities as the triples in the SEM set. The KG triples relate autoimmune diseases to specific genes, while the SEM triples associate pairs of rsIDs (SNPs) with psoriasis."

---

### Question 3: Multi-Disease Comparison (7 Duplicates Found)

**KG Triples (Sample):**
```
• (rs75851973 x rs13203895) --[ASSOCIATES_WITH]--> (Disease psoriasis)
• (Disease psoriasis) --[ASSOCIATES_WITH]--> (Gene HLA-B)
• (rs17728338 x rs13203895) --[ASSOCIATES_WITH]--> (Disease psoriasis)
```

**Semantic Triples (Sample):**
```
• (Disease Takayasu's arteritis) --[ASSOCIATES_WITH]--> (Gene ABHD16A)
• (Disease Takayasu's arteritis) --[ASSOCIATES_WITH]--> (Gene PRRC2A)
• (Disease psoriasis) --[ASSOCIATES_WITH]--> (Gene CTLA4)
```

**Duplicates Identified:**
```
• KG-2 = SEM-4: Both triples state an association between 'Disease psoriasis' and 'Gene HLA-B'. 
  Note: KG-2 reverses the order, but ASSOCIATES_WITH is implicitly bi-directional.
• KG-23 = SEM-24: Both state that Disease psoriasis is associated with Gene HLA-DRB1.
```

**LLM Deduplication Reasoning:**
> "The analysis identified triples across the KG and SEM datasets that express the same relationship (ASSOCIATES_WITH) between the same entities. KG-2 duplicates SEM-4 because ASSOCIATES_WITH is implicitly bidirectional."

---

### Question 6: Cystic Fibrosis (11 Duplicates Found - Highest)

**KG Triples (Sample):**
```
• (Disease cystic fibrosis) --[ASSOCIATES_WITH]--> (Gene CFTR)
• (Disease cystic fibrosis) --[PRESENTS]--> (Symptom Bronchiectasis)
• (Disease cystic fibrosis) --[ISA]--> (Disease autosomal recessive disease)
• (Disease cystic fibrosis) --[PRESENTS]--> (Symptom Abnormality of the liver)
```

**Semantic Triples (Sample):**
```
• (Disease cystic fibrosis) --[PRESENTS]--> (Symptom Pneumonia)
• (Disease cystic fibrosis) --[PRESENTS]--> (Symptom Nasal polyposis)
• (Disease cystic fibrosis) --[PRESENTS]--> (Symptom Cough)
```

**Duplicates Identified:**
```
• KG-3 = SEM-8: Both state cystic fibrosis presents with abnormality of the gastrointestinal tract.
• KG-5 = SEM-17: Both state cystic fibrosis presents with Bronchiectasis.
• KG-6 = SEM-3: Both state cystic fibrosis presents with cardiovascular abnormality.
```

**LLM Deduplication Reasoning:**
> "The analysis identified triples from the KG and SEM datasets that describe the same relationships between cystic fibrosis and its associated genes, symptoms, or classifications. Specifically, multiple symptom presentations were found to be duplicated."

---

### Question 17: Duchenne Muscular Dystrophy (6 Duplicates Found)

**KG Triples (Sample):**
```
• (Disease Duchenne muscular dystrophy) --[PRESENTS]--> (Symptom Muscle Weakness)
• (Disease Duchenne muscular dystrophy) --[ASSOCIATES_WITH]--> (Gene DMD)
• (Disease Duchenne muscular dystrophy) --[ISA]--> (Disease muscular dystrophy)
• (Disease Duchenne muscular dystrophy) --[ISA]--> (Disease X-linked recessive disease)
```

**Semantic Triples (Sample):**
```
• (Disease Duchenne muscular dystrophy) --[PRESENTS]--> (Symptom Cardiomyopathy)
• (Disease Duchenne muscular dystrophy) --[PRESENTS]--> (Symptom Cognitive impairment)
• (Disease Duchenne muscular dystrophy) --[PRESENTS]--> (Symptom Respiratory insufficiency)
```

**Duplicates Identified:**
```
• KG-1 = SEM-17: Both represent Duchenne muscular dystrophy presenting with Muscle Weakness.
• KG-7 = SEM-2: Both represent Duchenne muscular dystrophy presenting with Cardiomyopathy.
• KG-12 = SEM-1: Both represent Duchenne muscular dystrophy presenting with Behavioral abnormality.
```

---

## 🔄 LLM Deduplication Insights

### Types of Duplicates Found

1. **Exact Matches:** Same entities with same relationship (e.g., both say "X associates Y")
2. **Reversed Order:** Subject and object swapped, but same relationship (ASSOCIATES_WITH is bidirectional)
3. **Semantic Equivalence:** Different wording but same meaning (e.g., "Disease X" vs "X")

### Questions with Most Duplicates
| Question | Duplicates | Primary Type |
|----------|------------|--------------|
| q6_cystic_fibrosis | 11 | Symptom presentations |
| q3_multi_disease | 7 | Gene associations |
| q17_muscular_dystrophy | 6 | Symptom presentations |
| q11_inflammatory_bowel | 5 | Gene associations |
| q15_asthma_genetics | 5 | Gene associations |

### Questions with No Duplicates
- q1_hla_psoriasis (different entity types in each source)
- q2_diabetes_nod2 (variants vs gene associations)
- q7_parkinsons_genetics (no overlap)
- q10_schizophrenia (different variant sets)
- q16_colorectal_cancer (disease-specific vs syndrome-specific)
- q19_epilepsy (different epilepsy subtypes)

---

## 📈 Key Findings

### 1. SARG+ORAG Advantages
- **Highest average score** (9.86/10)
- **Won or tied on 16/20 questions**
- Best on comparison questions where both sources provide complementary information
- LLM deduplication successfully removed 57 redundant triples

### 2. Triple Source Analysis
- KG triples tend to be more structured (Gene-Disease associations)
- Semantic triples capture more symptoms and variant associations
- Combining both provides most comprehensive coverage

### 3. Deduplication Effectiveness
- Average 2.85 duplicates per question
- Symptom-related diseases have most duplicates (cystic fibrosis, muscular dystrophy)
- Variant-heavy questions have fewest duplicates (different rs IDs in each source)

---

## 🏁 Conclusion

The SARG+ORAG system with LLM-based deduplication achieves the best performance by:
1. Combining KG and semantic retrieval for comprehensive evidence
2. Extracting structured triples from both sources
3. Using LLM to intelligently identify and remove semantic duplicates
4. Multi-step reasoning with question analysis
5. Self-verification to ensure completeness

**Final Rankings:**
1. 🥇 SARG+ORAG: 9.86/10
2. 🥈 Vanilla RAG: 9.68/10  
3. 🥉 KG-RAG: 9.66/10

