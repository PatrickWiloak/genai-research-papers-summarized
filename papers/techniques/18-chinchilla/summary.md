# Training Compute-Optimal Large Language Models (Chinchilla)

**Authors:** Jordan Hoffmann, Sebastian Borgeaud, et al. (DeepMind)
**Published:** March 2022 (NeurIPS 2022)
**Paper:** [arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)

---

## Why This Matters

Chinchilla **rewrote the scaling laws** and changed how everyone trains LLMs:

- 🎯 **Proved GPT-3 was undertrained** - Needed 4× more data
- 📊 **New scaling ratio:** Equal scaling of params and tokens
- 💰 **Save billions:** Smaller models, more data = better & cheaper
- 🔥 **Validated by LLaMA:** Chinchilla laws proven in practice

**Real-world impact:**
- GPT-4: Trained following Chinchilla insights
- LLaMA: Explicitly used Chinchilla-optimal training
- Every major LLM since: Uses these scaling laws

**The insight:** Most models were **massively undertrained**. Train longer, not just bigger.

---

## The Problem with Original Scaling Laws

### OpenAI Scaling Laws (2020) Said:
```
For compute budget C:
Optimal N (params) ∝ C^0.73
Optimal D (tokens) ∝ C^0.27

Example at 1000× compute:
- Increase params by ~500×
- Increase data by ~10×
```

**Result:** GPT-3 (175B params, 300B tokens)

### Chinchilla Found This Was Wrong!

**Problem:**
- Models were too big, trained on too little data
- Better to use smaller models with more tokens
- GPT-3 should have been ~30B with 1.2T tokens

---

## New Scaling Laws

### Chinchilla's Discovery

```
For compute budget C:
Optimal N (params) ∝ C^0.50
Optimal D (tokens) ∝ C^0.50

Equal scaling! (vs OpenAI's 0.73/0.27)
```

**The Rule:** For every doubling of model size, double training tokens too.

### Practical Implications

| Compute Budget | Old Approach | Chinchilla-Optimal | Better? |
|----------------|--------------|-------------------|---------|
| 1× | 1B params, 20B tokens | 400M params, 8B tokens | ✅ |
| 10× | 10B params, 60B tokens | 2B params, 80B tokens | ✅ |
| 100× | 100B params, 200B tokens | 10B params, 2T tokens | ✅ |
| GPT-3 scale | 175B, 300B tokens | **70B, 1.4T tokens** | ✅ |

**Result:** Chinchilla 70B outperformed Gopher 280B

---

## Chinchilla vs Gopher

DeepMind's Comparison:

| Model | Params | Tokens | Performance | Training Cost |
|-------|--------|--------|-------------|---------------|
| Gopher | 280B | 300B | Baseline | 1.0× |
| **Chinchilla** | **70B** | **1.4T** | **Better** | **Same** |

**Key Finding:**
- Chinchilla: 4× smaller, trained on 5× more data
- Better performance on 67/68 benchmarks
- Same training compute!

---

## Methodology

### How They Found This

**Approach 1: Fix compute, vary N and D**
- Train many models with different N/D ratios
- Same total compute for all
- Find optimal balance

**Approach 2: IsoFLOP profiles**
- Measure loss for different N at fixed compute
- Find minimum loss point
- Repeat for different compute budgets

**Approach 3: Extrapolate from small models**
- Train small models extensively
- Fit scaling curves
- Predict optimal larger models

**All 3 methods agreed:** N and D should scale equally!

---

## Impact on the Field

### Before Chinchilla (2020-2022)
- GPT-3: 175B, 300B tokens
- Gopher: 280B, 300B tokens
- MT-NLG: 530B, 270B tokens
- **Trend:** Bigger and bigger

### After Chinchilla (2022+)
- Chinchilla: 70B, 1.4T tokens
- **LLaMA: 65B, 1.4T tokens** ✅
- **LLaMA 2: 70B, 2T tokens** ✅
- Mistral: 7B, heavily trained ✅
- **Trend:** Smaller models, more data

---

## Why This Matters for You

### Practical Takeaways

**If you have compute budget C:**
```python
# Old way (wrong):
params = large_number
tokens = small_number

# Chinchilla way (right):
params = sqrt(C / 6)  # Smaller model
tokens = sqrt(C / 6)  # More training data

# Both scale equally with compute
```

**Example:**
- Budget for 10B model on 100B tokens
- **Better:** 3B model on 1T tokens (10× more data)

### Cost Savings

**For same performance:**
- Chinchilla-optimal: 3× less inference cost (smaller model)
- Chinchilla-optimal: Same training cost
- **Win-win!**

---

## Real-World Validation

### LLaMA Proved It

**LLaMA 13B:**
- Trained on 1T tokens (Chinchilla-optimal)
- Matches GPT-3 175B
- **13× smaller!**

**LLaMA 65B:**
- Trained on 1.4T tokens
- Beats GPT-3 and Gopher
- Smaller and better

### Industry Adoption

**Everyone now uses Chinchilla scaling:**
- Meta: LLaMA series
- Mistral AI: All models
- Google: PaLM 2 (compute-optimal)
- Anthropic: Claude training

---

## Limitations

### 1. **Inference Cost vs Training Cost**
- Chinchilla optimizes for training
- Smaller model = cheaper inference
- But may sacrifice some capability

### 2. **Emergence May Need Size**
- Some capabilities emerge at larger sizes
- GPT-3's few-shot learning at 175B
- Smaller Chinchilla-optimal may miss these

### 3. **Dataset Availability**
- Need 4-5× more data than before
- High-quality data is limited
- May hit data walls

### 4. **Different Trade-offs for Deployment**
- If inference >> training cost
- May want larger model, less training
- Chinchilla optimal for research, not always production

---

## Key Formulas

### Compute Optimal Allocation

```
Given compute budget C (in FLOPs):

N_opt ≈ (C / 6)^0.50   # Optimal parameters
D_opt ≈ (C / 6)^0.50   # Optimal tokens

# Approximately:
N_opt ≈ 0.4 * C^0.50
D_opt ≈ 0.4 * C^0.50
```

### Loss Prediction

```
L(N, D) ≈ E + A/N^α + B/D^β

Where:
E ≈ 1.69  (irreducible loss)
A ≈ 406.4
B ≈ 410.7
α ≈ 0.34
β ≈ 0.28
```

---

## Comparison: Scaling Laws Evolution

| Aspect | Kaplan et al. 2020 | Chinchilla 2022 | What Changed |
|--------|-------------------|-----------------|--------------|
| **Param scaling** | C^0.73 | C^0.50 | Less aggressive |
| **Token scaling** | C^0.27 | C^0.50 | Much more! |
| **Ratio** | Favor params 3:1 | Equal 1:1 | Balanced |
| **GPT-3 verdict** | Optimal | Undertrained 4× | Big difference! |

---

## Key Takeaways

1. **Equal scaling:** Params and data should scale together (1:1 ratio)
2. **Most models undertrained:** GPT-3, Gopher, etc. needed 4× more data
3. **Smaller + more data:** Beats bigger + less data at same compute
4. **Validated in practice:** LLaMA proved Chinchilla was right
5. **Industry standard:** Everyone now trains Chinchilla-optimal

**Bottom line:** Chinchilla changed AI economics. Training smarter > training bigger.

---

## Further Reading

### Papers
- **Chinchilla:** https://arxiv.org/abs/2203.15556
- **Original Scaling Laws:** https://arxiv.org/abs/2001.08361
- **LLaMA (validation):** https://arxiv.org/abs/2302.13971

### Analysis
- **Epoch AI Blog:** Chinchilla implications
- **LessWrong:** Scaling laws deep dive
- **Gwern:** Scaling hypothesis

---

**Published:** March 2022
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - Rewrote how to train LLMs
**Citations:** 2000+
**Current Relevance:** Every major LLM uses these insights
**Legacy:** Changed AI from "bigger models" to "better training"
