# make_comparison_table.py
# AyShCXR — builds the model-comparison AUC table from per_disease_<arch>.csv files.
# Output: model_comparison_table.html (styled, professor format) + console markdown.
#
#   python make_comparison_table.py

import os, glob, pandas as pd

OUT_DIR = "/workspace"
DISEASE_ORDER = [
    "Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices","No Finding",
]
# pretty column names
NICE = {"efficientnet_b4":"EfficientNet-B4","densenet121":"DenseNet-121",
        "swin_t":"Swin-T","vit_b_16":"ViT-B/16","rad_dino":"Rad-DINO"}

cols = {}
for path in sorted(glob.glob(os.path.join(OUT_DIR, "per_disease_*.csv"))):
    arch = os.path.basename(path)[len("per_disease_"):-len(".csv")]
    s = pd.read_csv(path).set_index("disease")["auc"]
    cols[NICE.get(arch, arch)] = s

if not cols:
    print("No per_disease_*.csv files found. Train at least one model first."); raise SystemExit

tab = pd.DataFrame(cols).reindex(DISEASE_ORDER)
tab.loc["Mean AUC"] = tab.mean()                       # mean row at the bottom

# console (markdown)
print("\n=== CheXpert Plus — Model Comparison (val AUC) ===\n")
print(tab.round(4).to_markdown())

# styled HTML (green gradient = higher AUC, bold mean row)
styled = (tab.style
          .format("{:.4f}")
          .background_gradient(cmap="RdYlGn", axis=None, vmin=0.65, vmax=0.95)
          .set_caption("CheXpert Plus — Per-Disease & Mean Validation AUC by Architecture")
          .set_table_styles([{"selector":"caption","props":[("font-size","14px"),("font-weight","bold")]}]))
out = os.path.join(OUT_DIR, "model_comparison_table.html")
styled.to_html(out)
print(f"\nSaved styled table -> {out}")
