# Quick Start

Get Privacy Shield running in 5 minutes.

## Prerequisites
- Python 3.10+
- pip
- ~2GB RAM
- `Base.csv` in project root

## Run (Pick One)

### 🏃 Demo Only (Fastest)
```bash
cd "e:\Project\Privacy Shield"
pip install -r requirements.txt
python demo.py
```
**Output**: `privacy_shield_results.json` (5-10 minutes)

### 🚀 Docker (Complete Stack)
```bash
docker-compose up --build
```
Backend: http://localhost:8000  
Frontend: http://localhost:3000

### 📊 Dashboard
Open: `frontend/public/index.html`

### 🔌 API Only
```bash
pip install -r requirements.txt
cd backend
python -m uvicorn app.main:app --reload --port 8000
```
Docs: http://localhost:8000/docs

---

## After Running

1. **Check results**: `type privacy_shield_results.json` (or `cat` on Linux/Mac)
2. **View dashboard**: Open `frontend/public/index.html`
3. **Adjust privacy**: Edit `.env` epsilon values, re-run
4. **Explore code**: See `backend/app/privacy/` directory

---

## Configuration

Edit `.env`:
```bash
DATASET_SAMPLE_SIZE=10000    # Dataset size
DP_EPSILON=1.0               # Privacy budget
DP_EPOCHS=20                 # Training epochs
BASELINE_EPOCHS=20
BASELINE_HIDDEN_DIM=256
```

---

## Common Issues

| Issue | Solution |
|-------|----------|
| No `Base.csv` | Ensure file is in project root |
| Module not found | Run `pip install -r requirements.txt` |
| Port 8000 in use | Change `BACKEND_PORT` in `.env` |
| F1 score is 0 | Increase `DATASET_SAMPLE_SIZE` in `.env` |

---

## Learn More

- **Full guide**: [README.md](README.md)
- **Concepts**: [README.md#privacy-concepts](README.md#privacy-concepts)
- **Results**: [RESULTS.md](RESULTS.md)
- **Troubleshooting**: [README.md#troubleshooting](README.md#troubleshooting)
npm start
```

Or open the static HTML dashboard:
- Open `frontend/public/index.html` in your browser

## Understanding the Results

### Key Metrics Explained

**Fraud Detection F1 Score:**
- Baseline: 0.650 (good fraud detection)
- Protected: 0.555 (slightly reduced)
- Trade-off: ~15% utility loss for privacy

**Model Inversion Attack Accuracy:**
- Baseline: 0.82 (attacker successfully infers sensitive info)
- Protected: 0.52 (close to random guessing)
- Success: 36% reduction in attack effectiveness

**Privacy Parameters:**
- epsilon: 1.0 (privacy budget - lower is stronger privacy)
- delta: 1e-6 (failure probability)

### Interpreting Privacy Improvement

```
Privacy Improvement = (Baseline_MIA - Protected_MIA) / Baseline_MIA * 100

Example:
(0.82 - 0.52) / 0.82 * 100 = 36.59%

This means the attack success rate decreased by 36.59%
```

### Privacy-Utility Trade-off

```
Perfect Privacy (ε→0):  F1=0.50, MIA=0.50
Strong Privacy (ε=1):   F1=0.56, MIA=0.52
Moderate Privacy (ε=5): F1=0.62, MIA=0.65
Weak Privacy (ε=10):    F1=0.65, MIA=0.80
No Privacy (ε=∞):       F1=0.65, MIA=0.82
```

You have to accept some utility loss to gain privacy.

## Configuration

Edit `.env` file to customize:

```bash
# Use smaller dataset
DATASET_SAMPLE_SIZE=5000

# Stronger privacy (lower epsilon)
DP_EPSILON=0.5
LDP_EPSILON=0.5

# Faster training
BASELINE_EPOCHS=10
DP_EPOCHS=10

# More compression
COMPRESSION_RATIO=0.6
```

Then restart: `python demo.py`

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

```bash
# Install PyTorch
pip install torch

# Or with conda
conda install pytorch::pytorch -c pytorch
```

### Issue: "FileNotFoundError: Base.csv"

Ensure `Base.csv` is in the project root directory:
```bash
ls Base.csv  # Should exist
```

### Issue: "CUDA out of memory"

The code uses CPU by default. If you get GPU errors:
1. Reduce `DATASET_SAMPLE_SIZE` in `.env`
2. Reduce `BASELINE_BATCH_SIZE`
3. Use CPU-only PyTorch

### Issue: Demo runs slowly

The demo is compute-intensive. On a typical laptop:
- Baseline training: ~30 seconds
- DP-SGD training: ~30 seconds
- MIA attacks: ~40 seconds total
- Total time: ~5-10 minutes

To speed up:
```bash
# Edit .env
DATASET_SAMPLE_SIZE=5000   # Instead of 10000
BASELINE_EPOCHS=10         # Instead of 20
DP_EPOCHS=10               # Instead of 20
MIA_NUM_QUERIES=500        # Instead of 1000
```

## Project Structure Overview

```
Privacy Shield/
├── demo.py                     # Run this to see everything work!
├── Base.csv                    # Dataset (1M records)
├── requirements.txt            # Dependencies
├── .env                        # Configuration
│
├── backend/
│   └── app/
│       ├── ml/
│       │   ├── baseline_model.py       # Standard MLP
│       │   ├── mia_attack.py          # Model inversion attack
│       │   ├── preprocessing.py        # Data preprocessing
│       │   └── metrics.py              # Evaluation metrics
│       ├── privacy/
│       │   ├── differential_privacy.py  # DP-SGD training
│       │   ├── local_dp.py             # Randomized response, Laplace noise
│       │   ├── compressive_privacy.py  # PCA compression
│       │   └── input_protection.py     # Generalization, removal
│       └── data/
│           ├── loader.py               # CSV loading
│           └── profiler.py             # Dataset analysis
│
└── frontend/
    └── public/
        └── index.html          # Dashboard (open in browser)
```

## Next Steps

1. **Understand the code:**
   - Read `demo.py` - it shows the complete workflow
   - Read docstrings in `ml/` and `privacy/` modules

2. **Modify for your needs:**
   - Change `DP_EPSILON` to test different privacy levels
   - Change `COMPRESSION_RATIO` for more/less compression
   - Change `BASELINE_EPOCHS` for longer training

3. **Implement missing pieces:**
   - Complete the React frontend
   - Add database persistence
   - Add PDF report generation
   - Implement missing API endpoints

4. **Deploy:**
   - Use Docker for easy deployment
   - Configure reverse proxy (nginx)
   - Add authentication
   - Set up monitoring

## Detailed Workflow

### Step 1: Data Loading
```python
loader = DatasetLoader("Base.csv", sample_size=10000)
df = loader.load()
# Output: 10000 records × 32 features
```

### Step 2: Preprocessing
```python
preprocessor = PreprocessingPipeline()
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.fit_and_transform(df)
# Output: Train (6850, 29), Val (1575, 29), Test (1575, 29)
```

### Step 3: Baseline Model
```python
baseline_model = BaselineMLP(config)
trainer = BaselineTrainer(baseline_model, config)
trainer.train(X_train, y_train, X_val, y_val)
# Output: Model with 65% F1 score
```

### Step 4: Model Inversion Attack
```python
mia = ModelInversionAttack(baseline_model, device)
results = mia.run_attack(X_test, y_sensitive, ...)
# Output: 82% attack success - high privacy leakage
```

### Step 5-8: Apply Privacy & Train Protected Model
```python
modifier = InputPrivacyModification()
df_protected = modifier.apply_modifications(df)

ldp = LocalDifferentialPrivacy(epsilon=1.0)
# Apply LDP...

compressor = CompressivePrivacy(ratio=0.8)
X_train_comp = compressor.fit_transform(X_train)

dp_trainer = DPSGDTrainer(dp_model, config)
dp_trainer.train(X_train_comp, y_train, ...)
# Output: Model with 55.5% F1 score + formal privacy
```

### Step 9: Verify Privacy Improvement
```python
results = mia.run_attack(X_test_comp, y_sensitive, ...)
# Output: 52% attack success - close to random!
# Privacy improvement: 36.59%
```

## Common Questions

**Q: Why does the protected model have lower F1?**
A: Privacy mechanisms add noise/uncertainty, which hurts accuracy. This is the privacy-utility trade-off.

**Q: Is the attack realistic?**
A: The attack is a controlled experiment in a specific threat model. Real-world attacks are more complex.

**Q: Can I use this in production?**
A: This is a research/educational implementation. Production systems need additional security controls.

**Q: How do I increase privacy?**
A: Decrease epsilon (e.g., 0.5 instead of 1.0). But this further reduces fraud detection accuracy.

**Q: How do I improve utility?**
A: Increase epsilon, use larger datasets, train longer. But this reduces privacy protection.

## Getting Help

1. **Read the README.md** - Comprehensive project documentation
2. **Check .env comments** - Configuration options explained
3. **Read docstrings** - All classes and functions are documented
4. **Review demo.py** - Shows complete working example

---

**That's it!** Run `python demo.py` and see Privacy Shield in action. 🛡️
