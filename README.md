# Cognitive Load Estimation from EEG Signals for Programmers with Explainable AI and LLM Assistance

B.Sc. thesis, Department of Computer Science and Engineering, Pundra University of Science & Technology (PUST), Bogura, Bangladesh, 2026.

**Authors:** Nur E Tasfiya (0322310205101001), Mahabub A Elahi Rahat (0322310105101046)
**Supervisor:** Mst. Rehena Khatun, Assistant Professor, Dept. of CSE, PUB

---

## In one sentence

EEG detects **engagement** in code comprehension robustly, but detects the **difficulty level** of the code only weakly. This repository contains the experiments that show this, and a VS Code tool built on the signal that survived validation.

## What this thesis does

1. **Tests an assumption.** Workload models for programmers are often trained on generic EEG tasks and assumed to transfer to programming. We train a model on the STEW workload dataset and test it on real EEG from programmers reading Java code.
2. **Finds the signal that does work.** We train a programmer-specific EEG model, explain it with SHAP, and replicate the result on a second, independent programmer dataset.
3. **Tests the difficulty level.** On a dataset in which difficulty is controlled by design (confusing vs. clean code), we try four ways to detect how hard the code is.
4. **Builds a tool.** A VS Code extension and dashboard combine the validated EEG brain state with code complexity. A local LLM offers help only when the developer is engaged **and** the code is complex.

---

## Key results

| Question | Answer | Evidence |
|---|---|---|
| Does a STEW-trained model recognise workload in **new** people? | No | LOSO: 18 of 45 people correct = majority-class rate (40%), p = 0.56 |
| Why is its subject-mixed accuracy high (81.9 ± 2.0%)? | It recognises individuals | 10 person-identifying features alone reach 66.6% |
| Does the STEW model track programmers' code difficulty? | No | median ρ = 0.09, p = 0.63 (10 programmers) |
| Can EEG tell code reading from rest in an **unseen** programmer? | **Yes** | Programmer model, LOSO AUC 0.863 ± 0.045, 10 of 10, p = 0.002 |
| What does that model use? (SHAP) | Posterior theta, not eye-near channels | O2 theta strongest; posterior channels alone AUC 0.81 |
| Does an untrained θ/α load index agree? | **Yes, in two datasets** | Peitek 8/10; Atoms of Confusion 22/24, p = 0.0001 (30 of 34 programmers) |
| Does a single load index track difficulty? | No | Confusing vs. clean p = 0.68; rated difficult, wrong answers: no effect |
| Can a multi-feature classifier detect difficulty in unseen programmers? | **Weakly** | AUC 0.555, 19 of 24, p = 0.0006; reading time alone AUC 0.497 |
| Does time-locked EEG (FRP/ERP) do better? | No | FRP group effect replicated (17/24, p = 0.029), but single-trial AUC 0.53 (n.s.) |
| Does per-person calibration help? | No | AUC 0.535–0.541 vs. 0.556 without calibration |

### Correction of the mid-term results

The mid-term report claimed 82.59% accuracy and a significant improvement over the STEW baseline (t-test, p = 0.00013). **That claim is withdrawn.** The 82.59% was measured on people who also appeared in the training data, the t-test treated folds that share people as independent, and the STEW baseline (69%, support vector regression, Lim et al. 2018) was measured on nine people held out from training, so the two numbers are not comparable. Under leave-one-subject-out evaluation, our STEW model is at chance.

---

## Datasets

All three datasets are public and de-identified; no new human-subject data were recorded.

| Dataset | People | EEG | Task | Used for |
|---|---|---|---|---|
| **STEW** (Lim et al., IEEE TNSRE 2018) | 48 (45 rated), all male | Emotiv, 14 ch, 128 Hz | SIMKAP multitasking, 1–9 workload rating | Workload model |
| **Peitek et al.** (ESEC/FSE 2022) | 37 (we used 10) | LiveAmp, 64 ch, 500 Hz | 32 Java snippets, comprehension + rest | E1–E4, programmer model, tool demo |
| **Atoms of Confusion** (Bergum et al., Scientific Reports 2026) | 24 (9 female, 15 male) | 27 ch, 500 Hz | 36 confusing + 36 clean Java snippets | E5–E7 replication and difficulty level |

Download links:
- STEW: IEEE DataPort, doi:10.21227/44r8-ya50
- Peitek et al.: https://osf.io/4hjbd/ (raw data), https://github.com/brains-on-code/NoviceVsExpert (snippets)
- Atoms of Confusion: https://doi.org/10.5281/zenodo.14229849

Data are **not** included in this repository.

---

## Experiments

| ID | Question | Script |
|---|---|---|
| STEW | Subject-mixed vs. LOSO, fingerprint ablation, SHAP | `src/loso_eval.py`, `src/step2_ablation.py`, `src/shap_analysis.py` |
| E1, E2a | Does the STEW model transfer to programmers? | `run_participant.py`, `group_analysis.py` |
| E2b | Programmer model: code reading vs. rest (LOSO) | `rr_features.py`, `rr_analysis.py`, `rr_difficulty.py` |
| E2b SHAP | What the programmer model uses | `shap_programmer.py` |
| E3 | θ/α load index (Peitek) | `cl_index.py` |
| E4 | Behaviour vs. code complexity | `behav_analysis.py` |
| E5 | θ/α index on Atoms of Confusion | `aoc_index.py` |
| E6 | Difficulty-level classifier (112 features, XGBoost LOSO, reading time removed) | `aoc_level_ml.py`, `aoc_level_check.py`, `aoc_level_control.py` |
| E6 SHAP + calibration | What the level classifier uses; per-person calibration | `aoc_level_shap.py` |
| E7 | Fixation- and onset-locked potentials (FRP/ERP) | `aoc_frp_decode.py` |

---

## The tool

```
EEG playback (P05, unseen) ──P(engaged)──►┐
Code complexity (radon / lizard) ──level──►│  Flask backend  ├──► VS Code extension (status bar, warning, help)
Webcam eye tracker (optional) ──blink rate►│ (backend_v2.py) ├──► Browser dashboard (head map, timeline, BVI)
                                           └─────────────────┴◄─► Local LLM (Ollama, llama3.2, optional)
```

- **Brain state:** no headset was available, so the backend replays recorded EEG of participant P05, whom the programmer model never saw during training (one 5-s window every 5 s). A live headset would only change the data source.
- **Code level:** radon (Python) / lizard (Java) cyclomatic complexity, loops, conditions and Halstead effort, mapped to Low / Medium / High by experience profile (Beginner / Intermediate / Expert).
- **Decision rule:** the warning and the LLM suggestion appear **only when the brain state is Engaged and the code is High**. This is what separates the tool from always-on assistants.
- **Bug Vulnerability Index (BVI):** 0.35 × complexity + 0.35 × engagement (30-s mean) + 0.15 × SHAP + 0.15 × eye score. The weights are hand-set and **not validated** against real bug data; the BVI is a design concept.
- **Webcam eye tracker (experimental):** MediaPipe Face Landmarker + eye aspect ratio (EAR). A blink is an EAR drop below 75% of the user's own open-eye EAR for 50–500 ms; the blink rate of the last 60 s is sent every 5 s. Without a webcam the eye score is a neutral 0.3. Blink rate has no clear established relation to program comprehension (Peitek et al., EMIP 2018), so this module is not part of the evaluated results.

---

## Installation

Tested on Windows 11 with Python 3.14.

```bash
git clone https://github.com/Tasfiya-01/Real-Time-Cognitive-Load-Detection-for-Programmers.git
cd Real-Time-Cognitive-Load-Detection-for-Programmers

python -m venv venv
venv\Scripts\activate

pip install numpy scipy pandas matplotlib scikit-learn xgboost shap mne antropy statsmodels
pip install flask radon lizard requests joblib
pip install mediapipe opencv-python        # optional: webcam eye tracker

ollama pull llama3.2                       # optional: local LLM
code --install-extension vscode-extension/cognitive-load-detector-0.0.1.vsix
# if the .vsix is missing: cd vscode-extension && npx vsce package
```

## Running the tool

Use separate terminals and keep each one open.

```bash
# 1. (optional) local LLM
ollama serve

# 2. backend
python src/backend_v2.py

# 3. (optional) webcam blink rate, press Q to stop
python src/eye_tracking.py
```

Then:
- Open the dashboard at **http://localhost:5000/dashboard**. *Replay* shows P05's EEG and the snippet P05 was reading; *Live* analyses Python code you paste.
- Open any Python file in VS Code. The status bar shows the brain state and the code level, and the warning appears when both are high.

`models/` must contain `prog_model.pkl`, `playback_p05.npz`, `keep_idx.npy` and `snippets.json`. They are produced by `rr_analysis.py` and `make_playback.py`.

---

## Project structure

```
cognitive_load_thesis/
├── src/
│   ├── backend_v2.py              # Flask backend
│   ├── code_complexity.py         # radon CC, Halstead, loops/conditions
│   ├── personalized_threshold.py  # Beginner / Intermediate / Expert profiles
│   ├── eye_tracking.py            # webcam blink rate (MediaPipe + EAR)
│   ├── dashboard.html             # monitoring dashboard
│   ├── loso_eval.py, step2_ablation.py, shap_analysis.py   # STEW experiments
├── vscode-extension/              # VS Code extension
├── models/                        # programmer model, playback data, snippets
├── run_participant.py, group_analysis.py                    # E1, E2a
├── rr_features.py, rr_analysis.py, rr_difficulty.py         # E2b
├── shap_programmer.py                                        # E2b SHAP
├── cl_index.py, behav_analysis.py                            # E3, E4
├── aoc_index.py                                              # E5
├── aoc_level_ml.py, aoc_level_check.py, aoc_level_control.py # E6
├── aoc_level_shap.py                                         # E6 SHAP, calibration
├── aoc_frp_decode.py                                         # E7
├── make_playback.py               # playback file for the tool
└── results/                       # CSV outputs and figures
```

---

## Limitations

- The Peitek data have no workload ratings, so "load" in the programmer experiments is defined by task contrast (reading vs. rest) and by code properties.
- Reading and rest also differ in visual input. Posterior-only channels and SHAP reduce this concern but do not remove it.
- Only 10 of the 37 Peitek participants were used. Both programmer datasets come from the same research group and use short Java comprehension tasks, not writing or debugging.
- The difficulty-level signal is real but small (AUC 0.555) and is not usable for grading difficulty in a tool.
- The tool was demonstrated with replayed EEG, not a live headset, and without a user study. The BVI weights, the complexity thresholds and the blink-rate bands are hand-set.

## Future work

- Record EEG with a consumer headset during easy and hard programming tasks, with per-task workload ratings (e.g. NASA-TLX).
- Connect a live headset and run a user study of engagement-aware assistance.
- Combine fixation-locked and spectral EEG features, with more trials per person.
- Fit the BVI weights and the thresholds to real defect data.

---

## Main references

- W. L. Lim, O. Sourina, L. P. Wang, "STEW: Simultaneous task EEG workload data set," *IEEE TNSRE*, 26(11), 2106–2114, 2018.
- N. Peitek et al., "Correlates of programmer efficacy and their link to experience: A combined EEG and eye-tracking study," *ESEC/FSE*, 120–131, 2022.
- A. Bergum et al., "Fixation-related potentials reveal that confusing program code elicits a late frontal positivity," *Scientific Reports*, 16, 16833, 2026.
- N. Peitek et al., "Beyond gaze: Preliminary analysis of pupil dilation and blink rates in an fMRI study of program comprehension," *EMIP@ETRA*, 2018.
- T. Chen, C. Guestrin, "XGBoost: A scalable tree boosting system," *KDD*, 2016.
- S. M. Lundberg, S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS*, 2017.

The full reference list (44 entries) is in the thesis.

## License and data

Code: for academic use. Datasets remain under their original licences (STEW: CC BY; Peitek et al. stimuli: CC BY-SA 4.0; Atoms of Confusion: CC BY). Please cite the original dataset papers if you use them.
