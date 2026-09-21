# LinkedIn Participation Post Template

Copy and paste the text below for your official competition participation post on LinkedIn:

---

🚀 Excited to share our final methodology and solution for **The Pareidolia Paradox** Machine Learning Competition! 🌕

The challenge: Classify 256×256 lunar surface images into **Depth (class 0 / craters)** or **Rise (class 1 / mounds)** under variable solar illumination angles (`sun_azimuth_angle`), evaluated on Balanced Accuracy.

🔍 **The Core Discovery & Scientific Forensics:**
In lunar remote sensing, changing the sun's azimuth inverts shadow orientation, causing craters to look like mounds and mounds to look like craters — the classic "Pareidolia / Crater Illusion".
Through rigorous byte-level forensics, we uncovered:
1. **1,458 Duplicate Pairs in Train**: All size-2 duplicate pairs in the training data have pixel-identical images carrying conflicting labels under different azimuths. The label is therefore fundamentally determined by solar illumination geometry!
2. **Hidden Test Structural Decomposition**: The 2,000 test images cleanly separate into 3 distinct regimes:
   - **829 Train↔Test Overlaps (41.5%)**: Exact pixel matches to training images. Solved via a validated learned flip rule ($y_{test} = 1 - y_{train}$, 100% accurate on train duplicates).
   - **196 Intra-Test Duplicates (9.8%)**: 98 paired duplicates within test. Solved via our $lo\_\Delta$ transition model conditioned on $(az_{\text{lo}} \times \Delta az)$, achieving 0.8035 grouped CV Balanced Accuracy.
   - **975 Novel Images (48.8%)**: Resolved via an order-1 harmonic LightGBM model and Bayes-optimal step decision boundary at 270.0°, achieving ~0.7356 on our leakage-free distribution-matched proxy.

📈 **Results:**
- Honest regime-weighted validation estimate: **~0.8512** (distribution-matched proxy) / **~0.8747** (full-train OOF convention).
- 100% reproducible pipeline: raw data ➔ `train.py` ➔ serialized artifacts ➔ `inference.py` ➔ valid 2,000-row `submission.csv` in under 20 seconds.

💻 **Open Source Code & Methodology:**
- GitHub Repository: https://github.com/nikhilwankhedee/pareidolia-paradox
- Model Weights & Reproducibility Guide included!

Special thanks to the organizers for designing such a fascinating and intellectually rigorous competition!

#MachineLearning #DeepLearning #ComputerVision #DataScience #Kaggle #SpaceExploration #RemoteSensing #IEEE #AI
