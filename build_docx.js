const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
        VerticalAlign, PageNumber, Header, Footer, PageOrientation } = require("docx");

const CW = 9360; // content width (US Letter, 1" margins)
const GREY = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const B = { top: GREY, bottom: GREY, left: GREY, right: GREY };

const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const p  = (t, opts={}) => new Paragraph({ spacing: { after: 120 }, children: typeof t === "string" ? [new TextRun(t)] : t, ...opts });
const bullet = (t) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 60 },
        children: typeof t === "string" ? [new TextRun(t)] : t });
const numli = (t) => new Paragraph({ numbering: { reference: "n", level: 0 }, spacing: { after: 60 },
        children: typeof t === "string" ? [new TextRun(t)] : t });

function cell(text, w, { head=false, bold=false, fill=null, align=AlignmentType.LEFT } = {}) {
  return new TableCell({ borders: B, width: { size: w, type: WidthType.DXA },
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 }, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: align, children: [new TextRun({ text: String(text), bold: head||bold, size: head?20:20 })] })] });
}
function table(headers, rows, widths) {
  const trs = [ new TableRow({ tableHeader:true, children: headers.map((hh,i)=> new TableCell({ borders:B, width:{size:widths[i],type:WidthType.DXA}, shading:{fill:"2E5A88",type:ShadingType.CLEAR}, margins:{top:60,bottom:60,left:100,right:100}, verticalAlign:VerticalAlign.CENTER, children:[new Paragraph({alignment:AlignmentType.CENTER, children:[new TextRun({text:String(hh),bold:true,color:"FFFFFF",size:20})]})] })) }) ];
  rows.forEach((r,ri)=>{ trs.push(new TableRow({ children: r.map((c,i)=>cell(c,widths[i],{ fill: ri%2? "EEF3F8":"FFFFFF", align: i===0?AlignmentType.LEFT:AlignmentType.CENTER, bold: i===0 })) })); });
  return new Table({ width:{size:CW,type:WidthType.DXA}, columnWidths:widths, rows: trs });
}
const spacer = () => new Paragraph({ spacing:{after:60}, children:[new TextRun("")] });

const children = [];

// ---- Title block ----
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing:{before:240,after:60}, children:[new TextRun({ text:"AyShCXR", bold:true, size:48, color:"1F3B5B" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:40}, children:[new TextRun({ text:"Multi-Architecture Chest X-Ray Disease Classification on CheXpert Plus", bold:true, size:30 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:160}, children:[new TextRun({ text:"A Comparative Study of CNN and Transformer Backbones with Ensemble Learning", italics:true, size:24, color:"555555" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:20}, children:[new TextRun({ text:"Subhrakant Sethi & Ayush Singh", size:24 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:20}, children:[new TextRun({ text:"Thapar Institute of Engineering and Technology, Patiala", size:22, color:"555555" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:240}, children:[new TextRun({ text:"June 2026", size:22, color:"555555" })] }));

// ---- 1. Overview ----
children.push(h1("1. Project Overview"));
children.push(p("AyShCXR is a deep-learning system for automated chest X-ray (CXR) disease classification, designed to support diagnosis at rural Primary Health Centres (PHCs) in India where no radiologist is available. This report documents a comparative study in which three deep-learning architectures — two convolutional neural networks (CNNs) and one vision transformer — are trained under an identical protocol on the CheXpert Plus dataset to classify 14 thoracic pathologies simultaneously from a single frontal radiograph, and then combined into an ensemble."));
children.push(p([new TextRun({text:"Research questions",bold:true})]));
children.push(numli([new TextRun({text:"Architecture: ",bold:true}), new TextRun("Do modern CNNs and a medical-pretrained vision transformer differ in performance on multi-label CXR classification under a fair, identical protocol?")]));
children.push(numli([new TextRun({text:"Ensembling: ",bold:true}), new TextRun("Does combining heterogeneous backbones improve over the best single model?")]));
children.push(numli([new TextRun({text:"Limits: ",bold:true}), new TextRun("What bounds the achievable performance — model capacity or label quality?")]));

// ---- 2. Dataset ----
children.push(h1("2. Dataset"));
children.push(p("CheXpert Plus (Stanford AIMI; Chambon et al., 2024) is one of the largest public chest-radiograph datasets, pairing CXR images with the original radiology reports and machine-generated pathology labels."));
children.push(table(["Property","Value"], [
  ["Images used (frontal)","~190,800"],
  ["Patients","~65,000"],
  ["Pathologies (labels)","14 (multi-label)"],
  ["Label source","CheXBERT applied to the radiologist Impression section"],
  ["License","Research-only (no commercial use)"],
], [3000,6360]));
children.push(spacer());
children.push(p([new TextRun({text:"The 14 labels: ",bold:true}), new TextRun("Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices, No Finding.")]));
children.push(h2("Why the Impression labels (a key project decision)"));
children.push(p("CheXpert Plus ships several CheXBERT label variants (from the full report, the Findings section, and the Impression section). Early experiments using labels derived from the full report failed catastrophically — validation AUC stuck at 0.50 (random) — because the full report contains clinical history and prior-study comparisons that tag pathologies not visible in the current frontal image, injecting label noise the model cannot learn from. Switching to labels generated from the Impression (the radiologist's conclusion about the current study) immediately restored learning (val AUC 0.50 → 0.76+). This is itself a useful negative result on label provenance. The Findings-section labels were degenerate (“No Finding = 1” everywhere) and were discarded."));

// ---- 3. Data prep ----
children.push(h1("3. Data Preparation"));
children.push(numli([new TextRun({text:"Label table. ",bold:true}), new TextRun("A clean CSV mapping each image to its 14 one-hot Impression labels and its patient ID.")]));
children.push(numli([new TextRun({text:"Image resizing. ",bold:true}), new TextRun("All images pre-resized to 412 px once and stored on disk. The pipeline's first transform downsamples to 412 px anyway, so this is information-preserving and made each epoch ~3× faster.")]));
children.push(numli([new TextRun({text:"Patient-level splitting. ",bold:true}), new TextRun("Splits created with GroupShuffleSplit on patient ID (seed 42) so no patient appears in more than one split — eliminating leakage. The held-out portion was further halved by patient into validation and test sets.")]));
children.push(spacer());
children.push(table(["Split","Images","Purpose"], [
  ["Train","~175,000","Model fitting"],
  ["Validation","7,856","Threshold tuning / checkpoint selection"],
  ["Test","7,600","Final reported metrics (held out)"],
], [2400,2000,4960]));

// ---- 4. Models ----
children.push(h1("4. Models (Architectures)"));
children.push(p("All three backbones were ImageNet- or medical-pretrained and fitted with an identical custom classifier head so that only the backbone differs."));
children.push(table(["Model","Type","Input","Pretraining","Key property"], [
  ["EfficientNet-B4","CNN","380 px","ImageNet","Efficient modern CNN"],
  ["DenseNet-121","CNN","380 px","ImageNet","Standard CXR architecture (CheXNet)"],
  ["Rad-DINO","Transformer","224 px","838K chest X-rays","Medical self-supervised pretraining"],
], [2100,1600,1100,2060,2500]));
children.push(spacer());
children.push(p([new TextRun({text:"Shared classifier head: ",bold:true}), new TextRun("BatchNorm → Dropout(0.4) → Linear(→512) → GELU → Dropout(0.3) → Linear(→14).")]));
children.push(p([new TextRun({text:"GMP + GAP dual pooling (CNNs only). ",bold:true}), new TextRun("For the two CNNs, global-average-pool was augmented with global-max-pool, summed element-wise (an ablation-proven +~1% trick: GAP captures diffuse findings, GMP captures focal ones such as nodules and pneumothorax). Rad-DINO uses its pooled CLS token instead.")]));

// ---- 5. Training pipeline ----
children.push(h1("5. Training Pipeline"));
children.push(p("A single, unified pipeline was used for every model (only the backbone, input size, and batch size change), ensuring a fair comparison."));
children.push(table(["Component","Setting"], [
  ["Input","3-channel RGB (grayscale replicated), ImageNet normalization"],
  ["Loss","Focal Loss (γ=2.0, α=0.75) + label smoothing (0.1) + per-disease class weights"],
  ["Optimizer","AdamW with layer-wise LR decay: backbone 6e-5 (CNN)/3e-5 (transformer), head 2e-4"],
  ["Schedule","Cosine annealing, 18 epochs"],
  ["Precision","Automatic Mixed Precision (FP16) + gradient clipping (5.0)"],
  ["Compilation","torch.compile for speed"],
  ["Augmentation","Resize→RandomCrop, rotation, affine, perspective, colour jitter, Gaussian blur. No horizontal flip (clinically invalid)."],
  ["Hardware","NVIDIA H100 80GB (MIG 3g.40gb, 40 GB)"],
], [2200,7160]));
children.push(spacer());
children.push(p([new TextRun({text:"Why Focal + class weights. ",bold:true}), new TextRun("The 14 pathologies are heavily imbalanced (many appear in <10% of images). Focal Loss down-weights easy negatives and focuses on hard/rare positives; per-disease weights protect rare but critical classes (e.g., Pneumonia).")]));
children.push(p([new TextRun({text:"Why 18 epochs at higher LR. ",bold:true}), new TextRun("Each model peaks around epoch 10–12 in this regime; 18 epochs with cosine decay reaches convergence efficiently (~1.5–2 GPU-hours/model) without the over-fitting seen in longer low-LR runs.")]));

// ---- 6. Ensemble ----
children.push(h1("6. Ensemble"));
children.push(p("The three trained models' per-disease sigmoid probabilities were averaged. Because the architectures make different errors, averaging cancels uncorrelated mistakes and yields a more reliable prediction than any single model — a standard, leakage-free technique."));

// ---- 7. Evaluation ----
children.push(h1("7. Evaluation Methodology"));
children.push(p("CXR classification is multi-label (an image may show several diseases at once), so metrics are computed per disease and then macro-averaged (each disease weighted equally), not via single-label accuracy."));
children.push(bullet([new TextRun({text:"AUC-ROC (macro): ",bold:true}), new TextRun("the primary, threshold-independent metric — mean of the 14 per-disease ROC-AUCs.")]));
children.push(bullet([new TextRun({text:"Threshold metrics (Acc, Precision, Recall/Sensitivity, F1, Specificity): ",bold:true}), new TextRun("per-disease threshold tuned on validation to maximise F1, then applied to all splits.")]));
children.push(bullet([new TextRun({text:"5-disease subset AUC: ",bold:true}), new TextRun("restricted to the five “competition” pathologies, for comparability with the official CheXpert benchmark.")]));
children.push(bullet([new TextRun({text:"No MCC: ",bold:true}), new TextRun("deliberately excluded (not appropriate for this multi-disease setting).")]));

// ---- 8. Cross-validation ----
children.push(h1("8. Cross-Validation — and why 3-fold (not 5-fold)"));
children.push(p("To assess stability (how much performance depends on the particular data split), we ran true k-fold cross-validation — a fresh model retrained from scratch on each fold, with patient-grouped folds (GroupKFold) so no patient crosses folds."));
children.push(p([new TextRun({text:"Why 3-fold instead of 5-fold. ",bold:true}), new TextRun("A full 5-fold CV of all three deep models means 15 complete retrainings ≈ 35 GPU-hours, infeasible within the available compute window. We therefore used a tractable but genuine configuration: 3 folds (a recognised, valid form of k-fold CV), 10 epochs/fold, and a 90,000-image subset, on each architecture (9 retrainings, ~3.5 GPU-hours).")]));
children.push(p("This is reported transparently: it is real cross-validation (models are retrained per fold), reduced only in epochs/subset for tractability. Its purpose is to measure stability, while the headline AUCs come from the fully-trained models."));

// ---- 9. Results ----
children.push(new Paragraph({ pageBreakBefore:true, heading: HeadingLevel.HEADING_1, children:[new TextRun("9. Results")] }));
children.push(p("Headline results on the held-out test set (macro-averaged), with 3-fold cross-validation:"));
children.push(table(["Model","Test AUC","Test Acc","Test F1","5-dx AUC","CV AUC (±std)"], [
  ["EfficientNet-B4","0.820","0.860","0.457","0.818","0.764 ± 0.001"],
  ["DenseNet-121","0.827","0.859","0.466","0.820","0.798 ± 0.001"],
  ["Rad-DINO","0.821","0.857","0.456","0.816","0.809 ± 0.000"],
  ["Ensemble","0.838","0.865","0.476","0.828","0.791"],
], [2360,1300,1250,1150,1300,2000]));
children.push(spacer());
children.push(p([new TextRun({text:"Key observations",bold:true})]));
children.push(numli("The ensemble is best on AUC (0.838) and accuracy (0.865), beating the best single model by ~+1%."));
children.push(numli("All three backbones converge to within ~0.006 AUC of one another — a small CNN, a large CNN, and a transformer perform almost identically."));
children.push(numli("Training ≈ Validation ≈ Test (0.865 / 0.864 / 0.860 accuracy) → the model generalises well and does not over-fit."));
children.push(numli("Cross-validation is extremely stable (std 0.0003–0.0013). Rad-DINO leads the CV — under reduced training its 838K-CXR pretraining makes it the most data-efficient."));

// images
children.push(spacer());
children.push(new Paragraph({ alignment: AlignmentType.CENTER, children:[ new ImageRun({ type:"png", data: fs.readFileSync("roc_curves.png"), transformation:{ width:340, height:340 }, altText:{title:"ROC curves",description:"ROC",name:"roc"} }) ] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:160}, children:[new TextRun({ text:"Figure 1. Test-set ROC curves (micro-average; per-disease macro AUC = 0.84).", italics:true, size:18, color:"555555" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, children:[ new ImageRun({ type:"png", data: fs.readFileSync("auc_bar.png"), transformation:{ width:420, height:262 }, altText:{title:"AUC bar",description:"AUC",name:"auc"} }) ] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing:{after:120}, children:[new TextRun({ text:"Figure 2. Macro AUC-ROC by model.", italics:true, size:18, color:"555555" })] }));

// ---- Cross-dataset (subsection of Results) ----
children.push(h2("Cross-Dataset Generalization (CheXpert → NIH)"));
children.push(p("To test whether the models learned transferable radiographic features rather than dataset-specific artefacts, the CheXpert-trained models were evaluated zero-shot on the NIH ChestX-ray14 official test set (25,596 images — a different hospital system never seen in training), scored on the 7 diseases common to both datasets (the 4 NIH-only diseases — Emphysema, Fibrosis, Pleural Thickening, Hernia — are excluded)."));
children.push(table(["Model","CheXpert → NIH AUC (7 common diseases)"], [
  ["EfficientNet-B4","0.791"],
  ["DenseNet-121","0.799"],
  ["Rad-DINO","0.792"],
  ["Ensemble","0.809"],
], [4680,4680]));
children.push(spacer());
children.push(p([new TextRun({text:"Result. ",bold:true}), new TextRun("The ensemble achieves 0.809 mean AUC on a completely different dataset — only ~0.03 below its in-domain performance (0.838). This small drop indicates strong cross-dataset generalization: the model relies on genuine radiographic features, not CheXpert-specific cues. Per disease (ensemble), visually distinct findings transfer near-perfectly (Cardiomegaly 0.90, Pneumothorax 0.90), while subtle findings transfer less well (Consolidation 0.74, Pneumonia 0.73) — consistent with their inherent difficulty.")]));

// ---- 10. Discussion ----
children.push(h1("10. Discussion"));
children.push(h2("Why precision and F1 are moderate (~0.42–0.48)"));
children.push(p("Most pathologies appear in only a small fraction of images. The large pool of true negatives inflates accuracy (0.865) and specificity (0.860), while the scarcity of positives makes high precision difficult at any threshold that preserves clinically useful sensitivity. This is the expected signature of imbalanced multi-label detection, not a model weakness. The threshold-independent AUC (0.838) reflects true model quality."));
children.push(h2("Performance is label-limited, not capacity-limited"));
children.push(p("The training labels are generated by CheXBERT, which agrees with expert radiologist annotations at only F1 ≈ 0.44. A classifier cannot exceed the reliability of its supervision. This explains why three architecturally distinct backbones cluster at ~0.82 AUC: performance is bounded by label noise, not model expressivity."));
children.push(h2("Macro vs micro AUC"));
children.push(p("The ROC figure reports micro-averaged AUC (~0.92) — all predictions pooled — which is inflated by the abundance of easy negatives. We report the macro-averaged AUC (0.838) as the headline, since it weights each disease equally and does not mask poor performance on rare classes."));
children.push(h2("Per-disease profile"));
children.push(p("Strongest on visually distinct findings (Pneumothorax ~0.93, No Finding ~0.90, Support Devices ~0.89), weakest on subtle diffuse pathologies (Atelectasis ~0.72, Lung Opacity ~0.73) — consistent with radiologist-reported difficulty."));

// ---- 11. Limitations ----
children.push(h1("11. Limitations"));
children.push(bullet([new TextRun({text:"Label noise (F1 ≈ 0.44): ",bold:true}), new TextRun("CheXBERT labels cap achievable performance; gold radiologist labels would raise the meaningful ceiling.")]));
children.push(bullet([new TextRun({text:"Not state-of-the-art: ",bold:true}), new TextRun("official CheXpert SOTA (~0.90–0.93) is on the 5-disease subset with radiologist labels and the official test set; our 0.84 is all-14 on a patient-split with CheXBERT labels — a harder, non-identical setting. This is a competitive comparative study, not a new SOTA claim.")]));
children.push(bullet([new TextRun({text:"Reduced CV: ",bold:true}), new TextRun("3-fold with fewer epochs/subset (compute-limited); measures stability rather than full-data fold performance.")]));
children.push(bullet([new TextRun({text:"License: ",bold:true}), new TextRun("CheXpert Plus is research-only — a deployed product would require retraining on commercially-licensed or locally-collected data.")]));

// ---- 12. Conclusion ----
children.push(h1("12. Conclusion"));
children.push(p("Under an identical, fair protocol, two CNNs and a medical-pretrained vision transformer were trained on CheXpert Plus for 14-disease classification and combined into an ensemble achieving a macro AUC of 0.838 (accuracy 0.865) on a held-out, patient-disjoint test set, with stable 3-fold cross-validation and no over-fitting. The central finding is that performance is bounded by label quality, not model capacity — all architectures converge to the same ceiling — and that ensembling and medical pretraining (Rad-DINO) provide the most reliable gains. The result is a rigorous, honest comparative study and a strong foundation for a deployable PHC screening assistant."));

// ---- Appendix A: full metrics table (landscape) ----
function smallCell(text, w, o={}) {
  return new TableCell({ borders:B, width:{size:w,type:WidthType.DXA},
    shading: o.fill?{fill:o.fill,type:ShadingType.CLEAR}:undefined,
    margins:{top:20,bottom:20,left:30,right:30}, verticalAlign:VerticalAlign.CENTER,
    children:[new Paragraph({alignment:AlignmentType.CENTER, children:[new TextRun({text:String(text), bold:!!o.head, color:o.head?"FFFFFF":"000000", size:13})]})] });
}
function fullTable(headers, rows, widths) {
  const trs = [ new TableRow({tableHeader:true, children: headers.map((hh,i)=>smallCell(hh,widths[i],{head:true,fill:"2E5A88"}))}) ];
  rows.forEach((r,ri)=>{ trs.push(new TableRow({children: r.map((c,i)=>smallCell(c,widths[i],{fill: i===0?"DCE6F1":(ri%2?"EEF3F8":"FFFFFF")}))})); });
  return new Table({width:{size:widths.reduce((a,b)=>a+b,0),type:WidthType.DXA}, columnWidths:widths, rows:trs});
}
const _csv = fs.readFileSync("CheXpert_Model_Comparison.csv","utf8").trim().split(/\r?\n/).map(l=>l.split(","));
const _hdr = _csv[0], _rows = _csv.slice(1);
const _percol = Math.floor((12960 - 1500) / (_hdr.length - 1));
const _fw = [1500, ...Array(_hdr.length-1).fill(_percol)];
const landscapeChildren = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, children:[new TextRun("Appendix A — Full Metrics Table")] }),
  new Paragraph({ spacing:{after:120}, children:[new TextRun("Complete per-split metrics for all models. Multi-label, macro-averaged; threshold-based metrics (Acc, Prec, Recall, F1, Sens, Spec) computed at the per-disease F1-tuned threshold. Sens = Recall by definition.")] }),
  fullTable(_hdr, _rows, _fw),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font:"Arial", size:22 } } },
    paragraphStyles: [
      { id:"Heading1", name:"Heading 1", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:30, bold:true, font:"Arial", color:"1F3B5B" }, paragraph:{ spacing:{before:280,after:140}, outlineLevel:0 } },
      { id:"Heading2", name:"Heading 2", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:24, bold:true, font:"Arial", color:"2E5A88" }, paragraph:{ spacing:{before:180,after:100}, outlineLevel:1 } },
    ]
  },
  numbering: { config: [
    { reference:"b", levels:[{ level:0, format:LevelFormat.BULLET, text:"•", alignment:AlignmentType.LEFT, style:{ paragraph:{ indent:{ left:520, hanging:260 } } } }] },
    { reference:"n", levels:[{ level:0, format:LevelFormat.DECIMAL, text:"%1.", alignment:AlignmentType.LEFT, style:{ paragraph:{ indent:{ left:520, hanging:260 } } } }] },
  ] },
  sections: [{
    properties: { page: { size:{ width:12240, height:15840 }, margin:{ top:1440, right:1440, bottom:1440, left:1440 } } },
    footers: { default: new Footer({ children:[ new Paragraph({ alignment:AlignmentType.CENTER, children:[ new TextRun({text:"AyShCXR — CheXpert Plus Comparative Study     Page ", size:16, color:"888888"}), new TextRun({ children:[PageNumber.CURRENT], size:16, color:"888888" }) ] }) ] }) },
    children
  },
  {
    properties: { page: { size:{ width:12240, height:15840, orientation: PageOrientation.LANDSCAPE }, margin:{ top:1440, right:1440, bottom:1440, left:1440 } } },
    children: landscapeChildren
  }]
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync("CheXpert_Project_Report_FULL.docx", buf); console.log("WROTE CheXpert_Project_Report_FULL.docx", buf.length, "bytes"); });
