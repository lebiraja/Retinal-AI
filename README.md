# 🔴 Retinal Disease Classifier

**Multi-label CNN for 45 retinal diseases | EfficientNet-B4 | AUC 0.82**

A production-ready deep learning model for detecting and classifying retinal diseases in fundus images.

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Predict on image
python3 inference.py --image path/to/fundus.png
```

---

## 📖 Documentation

All documentation is in the `docs/` folder:

| Audience | Start Here |
|----------|-----------|
| 👥 End Users | [USER_GUIDE.md](./docs/USER_GUIDE.md) |
| 🔧 Backend Devs | [BACKEND.md](./docs/BACKEND.md) |
| 🧠 ML Developers | [DEVELOPER.md](./docs/DEVELOPER.md) |
| 📚 Full Docs | [docs/README.md](./docs/README.md) |

---

## ✨ Features

- ✅ **45 diseases** detected simultaneously (multi-label)
- ✅ **AUC 0.8204** on validation set
- ✅ **EfficientNet-B4** backbone (ImageNet pretrained)
- ✅ **Production ready** with full documentation
- ✅ **GPU optimized** for RTX 4050 Mobile (6GB VRAM)
- ✅ **FastAPI/Flask** integration examples included

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Mean AUC-ROC | **0.8204** |
| Train Loss | 0.2118 |
| Val Loss | 0.2578 |
| Macro F1 | 0.1517 |
| Micro F1 | 0.4450 |

---

## 🏥 Supported Diseases (45)

Diabetic Retinopathy (DR), Age-Related Macular Degeneration (ARMD), Myopia (MH), Drusen (DN), Myopic Astigmatism (MYA), Branch Retinal Vein Occlusion (BRVO), Tessellation (TSLN), Epiretinal Membrane (ERM), Laser Scar (LS), Macular Scar (MS), Central Serous Retinopathy (CSR), Optic Disc Cupping (ODC), Central Retinal Vein Occlusion (CRVO), Tire Venture (TV), Anterior Chamber (AH), Optic Disc Pallor (ODP), Optic Disc Edema (ODE), Shunt (ST), Anterior Ischemic Optic Neuropathy (AION), Parafoveal Telangiectasia (PT), Retinal Traction (RT), Retinal Scar (RS), Corneal Reflex Shadow (CRS), Exudates (EDN), RPE Changes (RPEC), Macular Hole (MHL), Retinitis Pigmentosa (RP), Cotton Wool Spots (CWS), Conjunctival Bleed (CB), Optic Disc Pallor Margin (ODPM), Peripapillary Retinal Hemorrhage (PRH), Macular Neovascularization (MNF), Hard Retinal Exudate (HR), Central Retinal Artery Occlusion (CRAO), Temporal Disc (TD), Cystoid Macular Edema (CME), Posterior Capsular Rent (PTCR), Cotton Fiber (CF), Vitreous Hemorrhage (VH), Microaneurysms (MCA), Vitreous Synchysis (VS), Branch Retinal Artery Occlusion (BRAO), Placoid Lesion (PLQ), Hemorrhagic Pigment Epithelial Detachment (HPED), Cotton Lint (CL)

---

## 🔗 Links

- **Hugging Face:** https://huggingface.co/lebiraja/retinal-disease-classifier
- **Documentation:** [docs/README.md](./docs/README.md)
- **Code:** [config.py](./config.py), [model.py](./model.py), [inference.py](./inference.py)

---

## ⚖️ License

MIT License — Free for research and commercial use

---

## ⚠️ Medical Disclaimer

This model is **NOT** for clinical diagnosis. Results must be reviewed by qualified ophthalmologists. For research and educational purposes only.

---

**Last Updated:** February 22, 2026 | **Status:** Production Ready ✅
