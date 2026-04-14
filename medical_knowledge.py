# medical_knowledge.py
# AyShCXR — AI Chest X-Ray Analysis System
# by Subhrakant Sethi & Ayush Singh
#
# v2 Updates:
#   ✅ More medical conditions added with disease-specific boosts
#   ✅ More medications added with pulmonary toxicity flags
#   ✅ Symptom combinations strengthened — multi-symptom = higher boost
#   ✅ History boost percentages refined — clinically validated ranges
#   ✅ New symptom combinations that narrow diagnosis more precisely

# ══════════════════════════════════════════════════════
# DISEASE INFORMATION DATABASE
# ══════════════════════════════════════════════════════

DISEASE_INFO = {

    "Atelectasis": {
        "full_name"       : "Atelectasis (Collapsed Lung Segment)",
        "icd10"           : "J98.11",
        "description"     : (
            "Atelectasis occurs when part or all of a lung "
            "collapses, reducing oxygen levels in the blood. "
            "Commonly seen after surgery, prolonged bed rest, "
            "or airway obstruction by mucus, tumour or foreign body."
        ),
        "imaging_findings": (
            "Increased opacity with ipsilateral mediastinal "
            "shift, elevated hemidiaphragm, displacement of "
            "fissures toward the collapsed area. No air bronchograms "
            "unlike consolidation."
        ),
        "common_symptoms" : [
            "Shortness of breath",
            "Rapid shallow breathing",
            "Dry cough",
            "Chest pain or tightness",
            "Low grade fever",
            "Cyanosis in severe cases",
            "Decreased breath sounds on affected side"
        ],
        "risk_factors"    : [
            "Recent surgery especially abdominal or chest",
            "Prolonged bed rest or immobility",
            "Smoking history",
            "Obesity",
            "Neuromuscular diseases",
            "Foreign body aspiration",
            "Mucus plugging in COPD or asthma",
            "Endobronchial tumour or lymph node compression"
        ],
        "severity"        : {
            "mild"  : "Small subsegmental atelectasis",
            "mod"   : "Lobar atelectasis",
            "severe": "Complete lung collapse"
        },
        "urgency"         : {
            "low"   : (0.35, 0.65),
            "medium": (0.65, 0.80),
            "high"  : (0.80, 1.00)
        },
        "urgency_message" : {
            "low"   : "Consult doctor within 48 hours",
            "medium": "Seek medical care today",
            "high"  : "Go to emergency room immediately"
        },
        "specialist"      : "Pulmonologist",
        "differential"    : [
            "Pneumonia — similar opacity but with fever and air bronchograms",
            "Pleural effusion — fluid not collapse, blunting of angle",
            "Pneumothorax — air not collapse, no lung markings",
            "Pulmonary mass — check for volume loss pattern"
        ],
        "lab_tests"       : [
            "Arterial blood gas (ABG)",
            "Complete blood count (CBC)",
            "Sputum culture if infection suspected",
            "CT chest if X-ray findings unclear",
            "Bronchoscopy if mucus plugging suspected"
        ],
        "treatments"      : [
            "Chest physiotherapy and deep breathing exercises",
            "Incentive spirometry to expand lungs",
            "Bronchodilators to open airways",
            "Oxygen therapy if SpO2 below 94%",
            "Treatment of underlying cause",
            "Early mobilisation after surgery",
            "Bronchoscopy for mucus plugging"
        ],
        "prescription_note": (
            "Bronchodilators like Salbutamol 2.5mg "
            "nebulisation 4-6 hourly may be prescribed. "
            "Chest physiotherapy is non-pharmacological "
            "and very effective. All treatment under "
            "medical supervision only."
        ),
        "follow_up"       : (
            "Repeat chest X-ray in 4-6 weeks to confirm "
            "resolution. If persistent, CT chest and "
            "bronchoscopy may be needed."
        ),
        "admission_criteria": (
            "Admit if SpO2 below 92% on room air, "
            "respiratory rate above 30/min, or complete "
            "lobar collapse causing respiratory distress."
        ),
        "prevention"      : [
            "Deep breathing exercises after surgery",
            "Early mobilisation after bed rest",
            "Incentive spirometry post-operatively",
            "Quit smoking before any surgery"
        ],
        "emergency_signs" : (
            "Seek emergency care if SpO2 drops below 90%, "
            "lips turn blue, or breathing becomes severely "
            "laboured."
        )
    },

    "Cardiomegaly": {
        "full_name"       : "Cardiomegaly (Enlarged Heart)",
        "icd10"           : "I51.7",
        "description"     : (
            "Cardiomegaly is an abnormally large heart on "
            "chest X-ray, defined as cardiothoracic ratio "
            "greater than 0.5. Always indicates underlying "
            "cardiac disease requiring full evaluation."
        ),
        "imaging_findings": (
            "Cardiothoracic ratio greater than 0.5. "
            "Globular cardiac silhouette suggests "
            "pericardial effusion. Left ventricular "
            "enlargement causes boot-shaped heart. "
            "Right ventricular enlargement in COPD."
        ),
        "common_symptoms" : [
            "Shortness of breath especially lying flat",
            "Swelling of legs and ankles",
            "Fatigue and weakness on exertion",
            "Irregular heartbeat or palpitations",
            "Dizziness or near fainting",
            "Chest pain or pressure",
            "Reduced exercise tolerance",
            "Waking at night breathless"
        ],
        "risk_factors"    : [
            "Hypertension — most common cause",
            "Coronary artery disease",
            "Diabetes mellitus",
            "Obesity",
            "Heavy alcohol consumption",
            "Family history of heart disease",
            "Previous heart attack",
            "Thyroid disease",
            "Anaemia",
            "Viral myocarditis",
            "Pregnancy — peripartum cardiomyopathy"
        ],
        "severity"        : {
            "mild"  : "CTR 0.50-0.55 — borderline",
            "mod"   : "CTR 0.55-0.65 — moderate",
            "severe": "CTR above 0.65 — severe"
        },
        "urgency"         : {
            "low"   : (0.35, 0.65),
            "medium": (0.65, 0.80),
            "high"  : (0.80, 1.00)
        },
        "urgency_message" : {
            "low"   : "Cardiology appointment within 1 week",
            "medium": "Seek cardiology evaluation this week",
            "high"  : "Urgent cardiac evaluation today"
        },
        "specialist"      : "Cardiologist",
        "differential"    : [
            "Pericardial effusion — fluid around heart, globular shape",
            "Left ventricular hypertrophy — concentric not dilated",
            "Dilated cardiomyopathy — thin walls, large chambers",
            "Valvular heart disease — murmur present",
            "Congenital heart disease — younger patient"
        ],
        "lab_tests"       : [
            "ECG — check for arrhythmia, LVH pattern",
            "Echocardiogram — essential for diagnosis",
            "BNP or NT-proBNP — heart failure marker",
            "Troponin — rule out acute MI",
            "Thyroid function tests",
            "Complete blood count",
            "Renal function tests",
            "Lipid profile"
        ],
        "treatments"      : [
            "ACE inhibitors — reduce heart workload",
            "Beta blockers — slow and strengthen beat",
            "Diuretics — remove excess fluid",
            "Digoxin — strengthen contractions",
            "Aldosterone antagonists",
            "ICD implantation for high risk patients",
            "Treat underlying cause aggressively"
        ],
        "prescription_note": (
            "Common medications: Enalapril 5-10mg twice "
            "daily, Carvedilol 3.125-25mg twice daily, "
            "Furosemide 20-80mg daily. All cardiac "
            "medications must be initiated by a cardiologist."
        ),
        "follow_up"       : (
            "Cardiology review every 3 months initially. "
            "Repeat echocardiogram every 6-12 months."
        ),
        "admission_criteria": (
            "Admit if acute decompensated heart failure, "
            "SpO2 below 94%, new arrhythmia, or "
            "suspected acute coronary syndrome."
        ),
        "prevention"      : [
            "Control blood pressure rigorously",
            "Manage diabetes and cholesterol",
            "Limit alcohol to safe limits",
            "Regular moderate exercise",
            "Quit smoking"
        ],
        "emergency_signs" : (
            "Call emergency for sudden severe chest pain, "
            "rapid irregular heartbeat, fainting, or "
            "acute breathlessness at rest."
        )
    },

    "Effusion": {
        "full_name"       : "Pleural Effusion (Fluid Around Lung)",
        "icd10"           : "J90",
        "description"     : (
            "Pleural effusion is abnormal fluid accumulation "
            "between the lung and chest wall. Always caused "
            "by an underlying condition requiring investigation. "
            "In India, TB and heart failure are the most common causes."
        ),
        "imaging_findings": (
            "Blunting of costophrenic angle (needs 200ml+). "
            "Homogeneous opacity in dependent areas. "
            "Meniscus sign on upright X-ray. "
            "Mediastinal shift away from large effusion."
        ),
        "common_symptoms" : [
            "Progressive shortness of breath",
            "Dry non-productive cough",
            "Sharp pleuritic chest pain",
            "Fever if infectious cause",
            "Chest heaviness on affected side",
            "Reduced breath sounds on auscultation",
            "Dullness on chest percussion"
        ],
        "risk_factors"    : [
            "Heart failure — most common cause in India",
            "Tuberculosis — very common in India",
            "Pneumonia — parapneumonic effusion",
            "Malignancy — lung, breast, lymphoma",
            "Liver cirrhosis — hepatic hydrothorax",
            "Nephrotic syndrome — low oncotic pressure",
            "Recent surgery or trauma",
            "Autoimmune diseases — lupus, RA"
        ],
        "severity"        : {
            "mild"  : "Small — less than 300ml",
            "mod"   : "Moderate — 300ml to 1500ml",
            "severe": "Large — more than 1500ml with mediastinal shift"
        },
        "urgency"         : {
            "low"   : (0.35, 0.65),
            "medium": (0.65, 0.80),
            "high"  : (0.80, 1.00)
        },
        "urgency_message" : {
            "low"   : "Consult doctor within 48 hours",
            "medium": "Seek medical care today",
            "high"  : "Emergency room immediately"
        },
        "specialist"      : "Pulmonologist or Thoracic Surgeon",
        "differential"    : [
            "Empyema — infected effusion needs urgent drainage",
            "Haemothorax — blood in pleural space after trauma",
            "Chylothorax — lymphatic fluid, milky appearance",
            "Atelectasis — opacity without costophrenic blunting"
        ],
        "lab_tests"       : [
            "Pleural fluid analysis — thoracentesis essential",
            "LDH and protein — exudate vs transudate (Light criteria)",
            "Pleural fluid culture and sensitivity",
            "ADA level — tuberculosis marker (above 40 U/L = TB)",
            "Pleural fluid cytology — malignancy",
            "BNP — heart failure marker",
            "Ultrasound chest — guide thoracentesis safely"
        ],
        "treatments"      : [
            "Thoracentesis — therapeutic drainage of fluid",
            "Treat underlying cause urgently",
            "Diuretics for cardiac cause",
            "Antibiotics for infectious cause",
            "Anti-TB treatment if tuberculosis confirmed",
            "Chest tube for large or recurrent effusion",
            "Pleurodesis for malignant recurrent effusion"
        ],
        "prescription_note": (
            "Treatment depends entirely on cause. "
            "Cardiac: Furosemide 40-80mg daily. "
            "Infectious: Antibiotics per culture. "
            "TB: Standard DOTS regimen (HRZE)."
        ),
        "follow_up"       : (
            "Repeat X-ray 4-6 weeks after treatment. "
            "If recurrent — CT chest and full malignancy "
            "workup needed urgently."
        ),
        "admission_criteria": (
            "Admit for large effusion causing respiratory "
            "distress, suspected empyema, or bilateral effusions."
        ),
        "prevention"      : [
            "Manage heart failure medications consistently",
            "Treat respiratory infections early and completely",
            "TB prevention — BCG vaccination, DOTS completion"
        ],
        "emergency_signs" : (
            "Emergency if sudden severe breathlessness, "
            "SpO2 below 90%, or tension physiology with "
            "mediastinal shift."
        )
    },

    "Infiltration": {
        "full_name"       : "Pulmonary Infiltration",
        "icd10"           : "J98.4",
        "description"     : (
            "Pulmonary infiltration refers to fluid, pus, "
            "cells or protein filling lung air spaces. "
            "Common causes include infection, inflammation, "
            "aspiration, and blood. In India, TB is the "
            "most important cause to exclude first."
        ),
        "imaging_findings": (
            "Hazy opacity without clear borders. "
            "Air bronchograms may be visible. "
            "May be unilateral or bilateral. "
            "Upper lobe infiltrates raise strong TB concern."
        ),
        "common_symptoms" : [
            "Persistent cough lasting more than 1 week",
            "Fever and chills",
            "Fatigue and weakness",
            "Shortness of breath on exertion",
            "Production of sputum",
            "Night sweats — particularly in TB",
            "Weight loss in chronic cases",
            "Haemoptysis in TB or malignancy"
        ],
        "risk_factors"    : [
            "Recent respiratory infection",
            "Immunocompromised state — HIV, steroids, diabetes",
            "Aspiration risk — swallowing difficulty",
            "Tuberculosis exposure or contact",
            "Smoking history",
            "Occupational dust exposure",
            "Diabetes mellitus",
            "Malnutrition"
        ],
        "severity"        : {
            "mild"  : "Focal subsegmental opacity",
            "mod"   : "Lobar or multilobar opacity",
            "severe": "Bilateral diffuse opacities"
        },
        "urgency"         : {
            "low"   : (0.35, 0.65),
            "medium": (0.65, 0.80),
            "high"  : (0.80, 1.00)
        },
        "urgency_message" : {
            "low"   : "Consult doctor within 48 hours",
            "medium": "Seek medical care today",
            "high"  : "Seek urgent care now"
        },
        "specialist"      : "Pulmonologist",
        "differential"    : [
            "Pneumonia — infection most likely cause",
            "Pulmonary TB — upper lobe, chronic symptoms",
            "Pulmonary oedema — bilateral butterfly pattern",
            "Pulmonary haemorrhage — acute onset, haemoptysis",
            "ARDS — bilateral rapid onset after trigger",
            "Interstitial lung disease — chronic, bilateral"
        ],
        "lab_tests"       : [
            "CBC with differential — neutrophilia = bacterial",
            "CRP and ESR — infection/inflammation markers",
            "Blood culture if febrile — before antibiotics",
            "Sputum AFB smear x3 — TB exclusion essential",
            "Sputum culture and sensitivity",
            "HIV test if risk factors present",
            "Mantoux or IGRA for TB",
            "Procalcitonin — bacterial vs viral"
        ],
        "treatments"      : [
            "Antibiotics for bacterial cause",
            "Anti-TB treatment if TB confirmed",
            "Antiviral for viral pneumonia",
            "Corticosteroids for inflammatory cause",
            "Oxygen therapy if SpO2 below 94%",
            "Chest physiotherapy",
            "Follow-up X-ray mandatory at 6 weeks"
        ],
        "prescription_note": (
            "Community acquired: Amoxicillin 500mg 3x daily "
            "for 5-7 days OR Azithromycin 500mg once daily. "
            "TB: Standard DOTS (HRZE 2 months then HR 4 months). "
            "Requires sputum culture before starting."
        ),
        "follow_up"       : (
            "Repeat X-ray at 4-6 weeks to confirm resolution. "
            "Non-resolving infiltration needs CT and bronchoscopy "
            "to exclude TB and malignancy."
        ),
        "admission_criteria": (
            "Admit if CURB-65 score 2 or more, SpO2 below "
            "94%, bilateral infiltrates, or suspected TB."
        ),
        "prevention"      : [
            "Influenza and pneumococcal vaccination",
            "BCG vaccination — TB prevention",
            "Good hand hygiene",
            "Avoid smoking and indoor air pollution"
        ],
        "emergency_signs" : (
            "Emergency if rapidly worsening breathlessness, "
            "coughing blood, confusion, or SpO2 below 90%."
        )
    },

    "Mass": {
        "full_name"       : "Pulmonary Mass (Lung Mass)",
        "icd10"           : "R91.8",
        "description"     : (
            "A pulmonary mass is an abnormal rounded opacity "
            "greater than 3cm. Requires urgent investigation "
            "to determine if benign or malignant. In smokers "
            "over 50, malignancy must be excluded first."
        ),
        "imaging_findings": (
            "Well or poorly defined rounded opacity greater "
            "than 3cm. Spiculated edges suggest malignancy. "
            "Satellite nodules, hilar lymphadenopathy, "
            "and pleural involvement are concerning features."
        ),
        "common_symptoms" : [
            "Persistent cough not resolving with treatment",
            "Haemoptysis — coughing up blood — red flag",
            "Progressive chest pain",
            "Unexplained weight loss more than 5kg",
            "Loss of appetite",
            "Fatigue and weakness",
            "Hoarse voice — recurrent laryngeal nerve",
            "Bone pain if metastatic disease",
            "Facial swelling — SVC syndrome"
        ],
        "risk_factors"    : [
            "Smoking — 85% of lung cancers",
            "Passive smoking exposure",
            "Radon gas exposure",
            "Asbestos occupational exposure",
            "Family history of lung cancer",
            "COPD or pulmonary fibrosis as background",
            "Air pollution — indoor and outdoor",
            "Age above 50 years"
        ],
        "severity"        : {
            "mild"  : "Benign appearing, well-defined, no features of malignancy",
            "mod"   : "Indeterminate features — PET CT needed",
            "severe": "Malignant features, lymphadenopathy, pleural involvement"
        },
        "urgency"         : {
            "low"   : (0.35, 0.65),
            "medium": (0.65, 0.80),
            "high"  : (0.80, 1.00)
        },
        "urgency_message" : {
            "low"   : "CT scan within 2 weeks",
            "medium": "Urgent CT scan this week",
            "high"  : "Immediate specialist evaluation today"
        },
        "specialist"      : "Pulmonologist + Oncologist",
        "differential"    : [
            "Primary lung cancer — most important to exclude",
            "Metastatic cancer — check for primary elsewhere",
            "Lung abscess — thick walled, air fluid level",
            "Carcinoid tumour — central, younger patient",
            "Pulmonary hamartoma — benign, popcorn calcification",
            "Fungal granuloma — endemic area exposure"
        ],
        "lab_tests"       : [
            "CT chest with contrast — urgent first step",
            "PET CT scan — metabolic activity assessment",
            "CT guided biopsy — tissue diagnosis",
            "Bronchoscopy with BAL and biopsy",
            "Sputum cytology x3",
            "Tumour markers — CEA, CYFRA 21-1, NSE",
            "Full staging workup — CT abdomen, bone scan"
        ],
        "treatments"      : [
            "Surgical resection if early stage and operable",
            "Chemotherapy for advanced disease",
            "Targeted therapy — EGFR, ALK, ROS1 mutations",
            "Immunotherapy — PD-L1 positive tumours",
            "Stereotactic body radiotherapy for inoperable",
            "Palliative care for advanced disease"
        ],
        "prescription_note": (
            "All treatment requires multidisciplinary team "
            "decision after tissue biopsy and molecular testing. "
            "No medication without confirmed histological diagnosis."
        ),
        "follow_up"       : (
            "CT chest at 1 month if biopsy pending. "
            "Annual CT surveillance if confirmed benign."
        ),
        "admission_criteria": (
            "Admit for haemoptysis requiring transfusion, "
            "respiratory failure, or urgent diagnostic workup."
        ),
        "prevention"      : [
            "Stop smoking — single most important action",
            "Avoid radon — test home in endemic areas",
            "Asbestos avoidance in occupation",
            "Low dose CT screening if high risk smoker over 50"
        ],
        "emergency_signs" : (
            "Immediate care for large haemoptysis, "
            "severe breathlessness, facial swelling, "
            "or SVC syndrome."
        )
    },

    "Nodule": {
        "full_name"       : "Pulmonary Nodule (Lung Nodule)",
        "icd10"           : "R91.1",
        "description"     : (
            "A pulmonary nodule is a small rounded opacity "
            "less than 3cm. Most are benign but some represent "
            "early lung cancer. Management depends on size, "
            "density, growth rate, and patient risk factors."
        ),
        "imaging_findings": (
            "Well-defined rounded opacity less than 3cm. "
            "Solid, ground glass or part-solid subtypes. "
            "Calcification in popcorn, laminar, or central "
            "pattern suggests benign aetiology."
        ),
        "common_symptoms" : [
            "Usually completely asymptomatic",
            "Found incidentally on X-ray for another reason",
            "Occasional dry cough if large or central"
        ],
        "risk_factors"    : [
            "Smoking — current or past",
            "Age above 35 years",
            "Family history of lung cancer",
            "Previous malignancy anywhere",
            "Occupational exposure to asbestos or radon",
            "COPD or emphysema as background",
            "TB exposure — granuloma common in India"
        ],
        "severity"        : {
            "mild"  : "Less than 6mm — low risk, no routine follow-up",
            "mod"   : "6-20mm — intermediate risk, serial CT",
            "severe": "Greater than 20mm — high risk, biopsy consideration"
        },
        "urgency"         : {
            "low"   : (0.35, 0.65),
            "medium": (0.65, 0.80),
            "high"  : (0.80, 1.00)
        },
        "urgency_message" : {
            "low"   : "CT scan follow-up in 3 months",
            "medium": "CT scan within 1 month",
            "high"  : "Specialist evaluation within 2 weeks"
        },
        "specialist"      : "Pulmonologist",
        "differential"    : [
            "Early lung cancer — must exclude first",
            "Metastatic deposit from other primary",
            "Carcinoid tumour — central location",
            "Hamartoma — benign, popcorn calcification",
            "Granuloma — TB or fungal, very common in India"
        ],
        "lab_tests"       : [
            "LDCT chest — Fleischner Society guidelines",
            "PET CT if nodule above 8mm or high risk patient",
            "CT guided biopsy if PET positive or growth on follow-up",
            "IGRA or Mantoux for TB aetiology in India"
        ],
        "treatments"      : [
            "Watchful waiting with serial CT scans",
            "CT at 3, 6, 12 months then annually",
            "PET CT if growth or suspicious features develop",
            "VATS resection if biopsy confirms malignancy",
            "SBRT radiotherapy if confirmed cancer but inoperable"
        ],
        "prescription_note": (
            "Most nodules require no medication. "
            "Antibiotic trial only if infectious cause likely. "
            "Anti-TB if Mantoux strongly positive."
        ),
        "follow_up"       : (
            "Fleischner Society guidelines: "
            "Less than 6mm — no routine follow-up if low risk. "
            "6-8mm — CT at 6-12 months. "
            "More than 8mm — CT at 3 months, then PET CT."
        ),
        "admission_criteria": (
            "Rarely requires admission. Admit only if "
            "urgent biopsy needed or complications develop."
        ),
        "prevention"      : [
            "Stop smoking immediately",
            "Annual LDCT screening if high risk smoker",
            "Avoid occupational carcinogens"
        ],
        "emergency_signs" : (
            "Seek urgent care if rapid growth on follow-up "
            "CT or haemoptysis develops from the nodule."
        )
    },

    "Pneumonia": {
        "full_name"       : "Pneumonia (Lung Infection)",
        "icd10"           : "J18.9",
        "description"     : (
            "Pneumonia is infection of the lung parenchyma "
            "causing consolidation of air spaces. Can be "
            "life threatening in elderly, diabetics, and "
            "immunocompromised patients. In India, "
            "Streptococcus pneumoniae is the most common cause."
        ),
        "imaging_findings": (
            "Lobar or segmental consolidation with "
            "air bronchograms. Bacterial: dense lobar. "
            "Atypical: bilateral patchy interstitial pattern. "
            "Klebsiella: upper lobe, bulging fissure sign."
        ),
        "common_symptoms" : [
            "High fever above 38.5 degrees Celsius",
            "Rigors and shaking chills",
            "Productive cough — yellow or green sputum",
            "Pleuritic chest pain worsening on breathing",
            "Shortness of breath",
            "Fatigue and malaise",
            "Confusion in elderly patients — key sign",
            "Rust coloured sputum in pneumococcal pneumonia"
        ],
        "risk_factors"    : [
            "Age above 65 or below 2 years",
            "Smoking history",
            "Alcohol excess",
            "Diabetes mellitus — Klebsiella risk",
            "Chronic lung disease",
            "Immunosuppression — HIV, steroids, chemotherapy",
            "Swallowing difficulty — aspiration risk",
            "Recent influenza infection",
            "Recent hospitalisation — nosocomial"
        ],
        "severity"        : {
            "mild"  : "CURB-65 score 0-1, treat outpatient",
            "mod"   : "CURB-65 score 2, consider hospital admission",
            "severe": "CURB-65 score 3-5, admit — ICU if 4 or 5"
        },
        "urgency"         : {
            "low"   : (0.35, 0.65),
            "medium": (0.65, 0.80),
            "high"  : (0.80, 1.00)
        },
        "urgency_message" : {
            "low"   : "Consult doctor within 24 hours",
            "medium": "Seek medical care today",
            "high"  : "Emergency room immediately"
        },
        "specialist"      : "General Physician or Pulmonologist",
        "differential"    : [
            "Pulmonary TB — chronic cough, night sweats, upper lobe",
            "Lung abscess — thick walled cavity with air fluid level",
            "Pulmonary oedema — bilateral butterfly, no fever",
            "Bronchial carcinoma — non-resolving on follow-up"
        ],
        "lab_tests"       : [
            "CBC — elevated WBC with neutrophilia = bacterial",
            "CRP and procalcitonin — severity markers",
            "Blood culture x2 before starting antibiotics",
            "Sputum Gram stain and culture",
            "Urine legionella and pneumococcal antigen",
            "ABG if SpO2 below 94% or respiratory distress"
        ],
        "treatments"      : [
            "Antibiotics — community vs hospital acquired protocol",
            "Oxygen therapy — target SpO2 94-98%",
            "Adequate IV hydration",
            "Antipyretics for fever above 38.5",
            "Chest physiotherapy and early mobilisation",
            "ICU admission if septic shock"
        ],
        "prescription_note": (
            "Community acquired mild-moderate: "
            "Amoxicillin 1g 3x daily for 5-7 days. "
            "If atypical suspected add Azithromycin 500mg. "
            "Diabetic patient: consider Klebsiella — use "
            "Ceftriaxone 1g IV. All require doctor prescription."
        ),
        "follow_up"       : (
            "Clinical review at 48-72 hours to confirm improvement. "
            "Repeat X-ray at 6 weeks mandatory to confirm resolution "
            "and exclude underlying malignancy."
        ),
        "admission_criteria": (
            "CURB-65 score 2 or more. SpO2 below 94%. "
            "Bilateral involvement. Sepsis criteria met. "
            "Failed outpatient treatment at 48 hours."
        ),
        "prevention"      : [
            "Pneumococcal vaccine — 13-valent then 23-valent",
            "Annual influenza vaccination",
            "Quit smoking",
            "Good hand hygiene and respiratory etiquette"
        ],
        "emergency_signs" : (
            "Emergency for severe breathlessness, "
            "SpO2 below 90%, confusion, or signs of septic shock."
        )
    },

    "Pneumothorax": {
        "full_name"       : "Pneumothorax (Collapsed Lung)",
        "icd10"           : "J93.9",
        "description"     : (
            "Pneumothorax is air in the pleural space "
            "causing lung collapse. Tension pneumothorax "
            "is immediately life threatening and requires "
            "emergency needle decompression."
        ),
        "imaging_findings": (
            "Visible pleural line with absent lung markings "
            "peripherally. Hyperlucent zone without vessels. "
            "Mediastinal shift AWAY from side indicates "
            "tension — radiological emergency."
        ),
        "common_symptoms" : [
            "Sudden sharp one-sided chest pain",
            "Abrupt onset breathlessness",
            "Rapid heart rate",
            "Decreased breath sounds on affected side",
            "Hypotension and raised JVP in tension",
            "Cyanosis in severe cases",
            "Tracheal deviation away from affected side"
        ],
        "risk_factors"    : [
            "Tall thin young male — primary spontaneous type",
            "Marfan syndrome — connective tissue disorder",
            "COPD or emphysema — secondary spontaneous type",
            "Severe asthma attack",
            "Lung cancer with pleural involvement",
            "Mechanical ventilation — barotrauma",
            "Recent chest procedures or trauma"
        ],
        "severity"        : {
            "mild"  : "Small less than 2cm rim — observe with oxygen",
            "mod"   : "Large more than 2cm — needle aspiration or drain",
            "severe": "Tension — emergency needle decompression first"
        },
        "urgency"         : {
            "low"   : (0.35, 0.65),
            "medium": (0.65, 0.80),
            "high"  : (0.80, 1.00)
        },
        "urgency_message" : {
            "low"   : "Seek medical care today — do not delay",
            "medium": "Emergency room now",
            "high"  : "Call emergency services immediately"
        },
        "specialist"      : "Emergency Physician / Thoracic Surgeon",
        "differential"    : [
            "Tension pneumothorax — tracheal shift, haemodynamic collapse",
            "Pulmonary embolism — similar sudden pain and breathlessness",
            "Myocardial infarction — central crushing pain",
            "Bullous emphysema — mimics pneumothorax on X-ray"
        ],
        "lab_tests"       : [
            "ABG — assess oxygenation urgently",
            "ECG — rule out MI as cause of chest pain",
            "CT chest if diagnosis clinically unclear",
            "Ultrasound — absence of lung sliding confirms diagnosis"
        ],
        "treatments"      : [
            "High flow oxygen 100% via non-rebreather mask",
            "Observation for small primary pneumothorax",
            "Needle aspiration for moderate primary",
            "Chest tube insertion for large or secondary",
            "VATS surgery for recurrent episodes",
            "TENSION: immediate 14G needle 2nd ICS MCL then drain"
        ],
        "prescription_note": (
            "MEDICAL EMERGENCY — no home treatment is safe. "
            "Tension pneumothorax: immediate needle at "
            "2nd intercostal space, midclavicular line."
        ),
        "follow_up"       : (
            "Repeat X-ray at 2-4 hours after aspiration. "
            "No flying for 6 weeks after complete resolution. "
            "No scuba diving ever after bilateral pneumothorax."
        ),
        "admission_criteria": (
            "All symptomatic pneumothorax. "
            "Secondary pneumothorax always admit. "
            "Tension pneumothorax — immediate ICU."
        ),
        "prevention"      : [
            "Quit smoking — reduces recurrence risk",
            "Avoid scuba diving if history of pneumothorax",
            "Surgical pleurodesis if recurrent — more than 2 episodes"
        ],
        "emergency_signs" : (
            "CALL AMBULANCE for sudden severe chest pain "
            "with worsening breathlessness, low blood pressure, "
            "or tracheal deviation — indicates tension pneumothorax."
        )
    },

    "Consolidation": {
        "full_name"       : "Pulmonary Consolidation",
        "icd10"           : "J98.4",
        "description"     : (
            "Consolidation occurs when air spaces fill "
            "with fluid, cells, or material. Most commonly "
            "caused by pneumonia but also blood, oedema, "
            "tumour, or organising pneumonia. "
            "TB must be excluded in Indian patients."
        ),
        "imaging_findings": (
            "Dense homogeneous opacity with air bronchograms. "
            "Lobar distribution suggests bacterial pneumonia. "
            "No volume loss unlike atelectasis. "
            "Upper lobe + cavitation = TB until proven otherwise."
        ),
        "common_symptoms" : [
            "Productive cough with purulent sputum",
            "High fever and rigors",
            "Pleuritic chest pain worsening on breathing",
            "Shortness of breath",
            "Dullness on percussion over affected area",
            "General malaise and weakness",
            "Night sweats if TB aetiology"
        ],
        "risk_factors"    : [
            "Bacterial respiratory infection",
            "Aspiration — swallowing difficulty",
            "Immunosuppression from any cause",
            "Recent influenza infection",
            "Smoking history",
            "Diabetes mellitus",
            "TB contact or exposure"
        ],
        "severity"        : {
            "mild"  : "Single lobe consolidation, well patient",
            "mod"   : "Multilobar consolidation, moderate symptoms",
            "severe": "Bilateral consolidation, ARDS, respiratory failure"
        },
        "urgency"         : {
            "low"   : (0.35, 0.65),
            "medium": (0.65, 0.80),
            "high"  : (0.80, 1.00)
        },
        "urgency_message" : {
            "low"   : "Consult doctor within 24 hours",
            "medium": "Seek medical care today",
            "high"  : "Emergency room immediately"
        },
        "specialist"      : "Pulmonologist",
        "differential"    : [
            "Bacterial pneumonia — most common cause",
            "Pulmonary TB — upper lobe, chronic, night sweats",
            "Lung infarction — post pulmonary embolism",
            "Bronchioloalveolar carcinoma — slow growing",
            "Organising pneumonia — not resolving on antibiotics"
        ],
        "lab_tests"       : [
            "CBC with differential",
            "CRP and procalcitonin",
            "Blood cultures x2 before antibiotics",
            "Sputum AFB x3 — TB must be excluded",
            "Sputum culture and sensitivity",
            "ABG if hypoxic",
            "CT chest and bronchoscopy if non-resolving"
        ],
        "treatments"      : [
            "Antibiotics guided by clinical setting and culture",
            "Oxygen therapy as needed to maintain SpO2",
            "Chest physiotherapy and hydration",
            "NSAIDs for pleuritic pain",
            "Anti-TB if TB confirmed",
            "Follow-up X-ray at 6 weeks mandatory"
        ],
        "prescription_note": (
            "Empirical: Amoxicillin-clavulanate 625mg 3x daily "
            "plus Azithromycin 500mg once daily for 7 days. "
            "If TB suspected: do NOT start antibiotics alone — "
            "refer for full TB workup first."
        ),
        "follow_up"       : (
            "Repeat X-ray at 6 weeks mandatory. "
            "Non-resolving consolidation needs CT chest "
            "and bronchoscopy to exclude malignancy and TB."
        ),
        "admission_criteria": (
            "Multilobar consolidation, SpO2 below 94%, "
            "sepsis criteria, or suspected TB requiring isolation."
        ),
        "prevention"      : [
            "Pneumococcal and influenza vaccination",
            "Quit smoking",
            "Good oral hygiene to prevent aspiration"
        ],
        "emergency_signs" : (
            "Emergency for bilateral consolidation, "
            "SpO2 below 90%, or septic shock presentation."
        )
    },

    "Edema": {
        "full_name"       : "Pulmonary Oedema (Fluid in Lungs)",
        "icd10"           : "J81.1",
        "description"     : (
            "Pulmonary oedema is excess fluid in lung tissue "
            "from raised pulmonary venous pressure or capillary "
            "leak. Cardiogenic shows butterfly perihilar pattern. "
            "A medical emergency requiring immediate treatment."
        ),
        "imaging_findings": (
            "Perihilar butterfly pattern in cardiogenic. "
            "Upper lobe vascular diversion — early sign. "
            "Kerley B lines — horizontal lines at lung bases. "
            "Usually with cardiomegaly if cardiac cause."
        ),
        "common_symptoms" : [
            "Acute severe breathlessness — sudden onset",
            "Orthopnoea — cannot lie flat at all",
            "Paroxysmal nocturnal dyspnoea — waking at night",
            "Coughing pink frothy sputum — severe sign",
            "Wheeze — cardiac asthma",
            "Cold clammy skin and profuse sweating",
            "Cyanosis of lips and fingertips"
        ],
        "risk_factors"    : [
            "Heart failure — most common cause",
            "Hypertensive emergency",
            "Acute myocardial infarction",
            "Severe mitral or aortic valve disease",
            "Fluid overload — renal failure or excessive IV fluids",
            "Sepsis — non-cardiogenic ARDS pattern",
            "Inhalation injury — toxic gas exposure"
        ],
        "severity"        : {
            "mild"  : "Kerley B lines, upper lobe diversion only",
            "mod"   : "Perihilar opacities, mild hypoxia, SpO2 90-94%",
            "severe": "Bilateral opacities, frothy sputum, SpO2 below 90%"
        },
        "urgency"         : {
            "low"   : (0.35, 0.65),
            "medium": (0.65, 0.80),
            "high"  : (0.80, 1.00)
        },
        "urgency_message" : {
            "low"   : "Seek medical care today",
            "medium": "Emergency room now",
            "high"  : "Call ambulance immediately — life threatening"
        },
        "specialist"      : "Cardiologist / Emergency Physician",
        "differential"    : [
            "Pneumonia — fever, usually unilateral",
            "ARDS — bilateral, non-cardiogenic trigger",
            "Bilateral pleural effusion — no air space opacity"
        ],
        "lab_tests"       : [
            "BNP or NT-proBNP — cardiogenic vs non-cardiogenic",
            "Troponin — acute MI as precipitant",
            "ECG — arrhythmia, ischaemia, LVH",
            "Echocardiogram — urgent bedside if available",
            "ABG — severity of hypoxia",
            "Renal function and electrolytes"
        ],
        "treatments"      : [
            "Sit upright — do not lie flat",
            "High flow oxygen 100% via non-rebreather mask",
            "IV Furosemide 40-80mg stat",
            "IV Glyceryl trinitrate infusion if BP above 100",
            "CPAP or BiPAP if not improving on oxygen",
            "Intubation and ventilation if failing"
        ],
        "prescription_note": (
            "ACUTE EMERGENCY — hospital treatment only. "
            "IV Furosemide 40-80mg, IV GTN infusion. "
            "Daily weight monitoring mandatory at home. "
            "Salt restriction less than 2g per day."
        ),
        "follow_up"       : (
            "Daily weight monitoring at home. "
            "Cardiology review every 1-3 months. "
            "Echocardiogram every 6 months minimum."
        ),
        "admission_criteria": (
            "All acute pulmonary oedema must be admitted. "
            "ICU admission if intubation needed."
        ),
        "prevention"      : [
            "Take all heart failure medications every day without fail",
            "Restrict salt intake to less than 2g per day",
            "Weigh daily — gain of 2kg in 2 days means call doctor"
        ],
        "emergency_signs" : (
            "CALL AMBULANCE — pink frothy sputum, "
            "unable to breathe lying flat, losing consciousness."
        )
    },

    "Emphysema": {
        "full_name"       : "Emphysema (COPD)",
        "icd10"           : "J43.9",
        "description"     : (
            "Emphysema is permanent destruction of alveolar "
            "walls causing air trapping and hyperinflation. "
            "Caused almost exclusively by smoking in adults. "
            "Alpha-1 antitrypsin deficiency in young patients."
        ),
        "imaging_findings": (
            "Hyperinflated lungs — more than 6 anterior "
            "ribs visible on PA film. Flattened hemidiaphragms. "
            "Barrel chest appearance. Bullae — avascular areas "
            "at lung apices. Increased retrosternal space."
        ),
        "common_symptoms" : [
            "Gradual progressive breathlessness over years",
            "Chronic cough — usually morning",
            "Wheeze on exertion",
            "Pursed lip breathing at rest",
            "Use of accessory muscles of breathing",
            "Weight loss in advanced disease",
            "Barrel chest appearance on examination"
        ],
        "risk_factors"    : [
            "Smoking — 85-90% of cases",
            "Passive smoking exposure for many years",
            "Alpha-1 antitrypsin deficiency — genetic",
            "Occupational dust and fumes — coal, cotton",
            "Indoor air pollution — biomass fuel cooking",
            "Age above 40 with smoking history"
        ],
        "severity"        : {
            "mild"  : "FEV1 above 80% predicted — GOLD stage 1",
            "mod"   : "FEV1 50-79% predicted — GOLD stage 2",
            "severe": "FEV1 below 50% predicted — GOLD stage 3-4"
        },
        "urgency"         : {
            "low"   : (0.35, 0.65),
            "medium": (0.65, 0.80),
            "high"  : (0.80, 1.00)
        },
        "urgency_message" : {
            "low"   : "Pulmonologist appointment within 1 week",
            "medium": "Pulmonology evaluation this week",
            "high"  : "Urgent respiratory evaluation today"
        },
        "specialist"      : "Pulmonologist",
        "differential"    : [
            "Chronic bronchitis — productive cough predominant",
            "Asthma — reversible airflow limitation, younger",
            "Bronchiectasis — recurrent infections, copious sputum",
            "Heart failure — bilateral basal crackles, oedema"
        ],
        "lab_tests"       : [
            "Spirometry — FEV1/FVC ratio below 0.7 confirms COPD",
            "Full lung function tests including DLCO",
            "ABG — assess oxygenation and CO2 retention",
            "Alpha-1 antitrypsin level if under 45 or non-smoker",
            "CBC — secondary polycythaemia from hypoxia",
            "HRCT chest — quantify emphysema extent"
        ],
        "treatments"      : [
            "Stop smoking — single most important intervention",
            "Short acting bronchodilator SABA as needed",
            "Tiotropium 18mcg once daily — long acting",
            "ICS-LABA combination if frequent exacerbations",
            "Pulmonary rehabilitation program — proven benefit",
            "Long term oxygen therapy if PaO2 below 55mmHg"
        ],
        "prescription_note": (
            "Salbutamol MDI 100mcg 2 puffs as needed. "
            "Tiotropium 18mcg once daily via Handihaler. "
            "Correct inhaler technique is critical — "
            "ask pharmacist to demonstrate."
        ),
        "follow_up"       : (
            "Spirometry every 12 months to monitor decline. "
            "Annual influenza vaccination mandatory. "
            "5 yearly pneumococcal vaccination."
        ),
        "admission_criteria": (
            "Acute exacerbation with SpO2 below 90%, "
            "inability to manage at home, or cor pulmonale."
        ),
        "prevention"      : [
            "Never smoke or quit immediately — most important",
            "Avoid all secondhand smoke exposure",
            "Annual influenza vaccine every year",
            "Use N95 mask if exposed to occupational dust"
        ],
        "emergency_signs" : (
            "Emergency for acute severe worsening breathlessness, "
            "SpO2 below 88%, or central cyanosis of lips."
        )
    },

    "Fibrosis": {
        "full_name"       : "Pulmonary Fibrosis (Scarred Lung)",
        "icd10"           : "J84.10",
        "description"     : (
            "Pulmonary fibrosis is progressive scarring of "
            "lung tissue. Irreversible — progression can only "
            "be slowed not reversed. Median survival without "
            "treatment is 2-5 years from diagnosis."
        ),
        "imaging_findings": (
            "Bilateral basal subpleural reticular opacities. "
            "Honeycombing with traction bronchiectasis. "
            "HRCT shows UIP pattern — diagnostic. "
            "Upper lobe predominance suggests HP or sarcoidosis."
        ),
        "common_symptoms" : [
            "Gradual progressive breathlessness on exertion",
            "Dry persistent non-productive cough",
            "Fatigue and reduced exercise tolerance",
            "Weight loss in advanced disease",
            "Clubbing of fingers — 50% of IPF cases",
            "Velcro-like crackles at lung bases on auscultation"
        ],
        "risk_factors"    : [
            "Age above 60 — IPF rare below 50",
            "Male gender — IPF more common in men",
            "Smoking history — significant risk factor",
            "Bird and animal exposures — hypersensitivity pneumonitis",
            "Occupational dust — silica, asbestos, coal",
            "Connective tissue diseases — RA, scleroderma",
            "Medications — nitrofurantoin, amiodarone, methotrexate",
            "Gastro-oesophageal reflux — aspiration ILD"
        ],
        "severity"        : {
            "mild"  : "Mild restriction FVC above 70%, DLCO above 60%",
            "mod"   : "Moderate FVC 50-70%, DLCO 40-60%",
            "severe": "Severe FVC below 50%, DLCO below 40%"
        },
        "urgency"         : {
            "low"   : (0.35, 0.65),
            "medium": (0.65, 0.80),
            "high"  : (0.80, 1.00)
        },
        "urgency_message" : {
            "low"   : "Pulmonologist within 2 weeks",
            "medium": "ILD specialist this week",
            "high"  : "Urgent ILD specialist evaluation"
        },
        "specialist"      : "Pulmonologist (ILD Specialist)",
        "differential"    : [
            "Hypersensitivity pneumonitis — antigen exposure history",
            "NSIP — younger, connective tissue disease association",
            "Sarcoidosis — bilateral hilar lymphadenopathy",
            "Drug induced ILD — medication history essential",
            "Asbestosis — occupational history, pleural plaques"
        ],
        "lab_tests"       : [
            "HRCT chest — most important investigation",
            "Full lung function tests — FVC, TLC, DLCO",
            "ANA, ANCA, anti-CCP — connective tissue screen",
            "Bronchoscopy with BAL — cell count pattern",
            "Surgical lung biopsy if diagnosis unclear on HRCT",
            "6 minute walk test — functional assessment",
            "Echocardiogram — pulmonary hypertension screening"
        ],
        "treatments"      : [
            "Pirfenidone — antifibrotic, slows FVC decline",
            "Nintedanib — alternative antifibrotic agent",
            "Pulmonary rehabilitation program",
            "Long term oxygen if resting SpO2 below 88%",
            "Lung transplant assessment in eligible patients"
        ],
        "prescription_note": (
            "Pirfenidone 267mg 3x daily with meals — "
            "titrate to 801mg 3x daily over 3 weeks. "
            "Nintedanib 150mg twice daily with food. "
            "Both require specialist initiation and monitoring."
        ),
        "follow_up"       : (
            "Lung function tests every 3-6 months. "
            "HRCT every 12 months. "
            "Monitor closely for acute exacerbation of IPF."
        ),
        "admission_criteria": (
            "Acute exacerbation of IPF — bilateral "
            "ground glass on background of UIP pattern."
        ),
        "prevention"      : [
            "Remove antigen source immediately if HP",
            "Quit smoking",
            "Treat GORD aggressively in ILD patients",
            "Avoid bird and mold exposures"
        ],
        "emergency_signs" : (
            "Acute exacerbation of IPF is life threatening. "
            "Emergency for sudden significant worsening of "
            "breathlessness over days in known fibrosis."
        )
    },

    "Pleural Thickening": {
        "full_name"       : "Pleural Thickening",
        "icd10"           : "J92.9",
        "description"     : (
            "Pleural thickening is fibrous scarring of the "
            "pleural lining. Common causes include previous "
            "infection particularly TB, trauma, and asbestos "
            "exposure. Bilateral calcified plaques = asbestos."
        ),
        "imaging_findings": (
            "Linear opacity along the chest wall. "
            "May be unilateral or bilateral. "
            "Calcification suggests old TB or asbestos exposure. "
            "Apical pleural capping common after TB."
        ),
        "common_symptoms" : [
            "Breathlessness on exertion if significant thickening",
            "Chest tightness or dull ache",
            "Reduced exercise tolerance",
            "Often completely asymptomatic if mild"
        ],
        "risk_factors"    : [
            "Previous pulmonary TB — very common in India",
            "Asbestos occupational exposure",
            "Previous pleural infection or empyema",
            "Haemothorax after trauma",
            "Previous thoracic surgery",
            "Connective tissue diseases",
            "Radiation therapy to chest"
        ],
        "severity"        : {
            "mild"  : "Less than 5mm — usually asymptomatic",
            "mod"   : "5-10mm with some restrictive impairment",
            "severe": "More than 10mm bilateral with restriction"
        },
        "urgency"         : {
            "low"   : (0.35, 0.65),
            "medium": (0.65, 0.80),
            "high"  : (0.80, 1.00)
        },
        "urgency_message" : {
            "low"   : "Routine pulmonology review",
            "medium": "Pulmonologist within 2 weeks",
            "high"  : "Specialist evaluation this week"
        },
        "specialist"      : "Pulmonologist",
        "differential"    : [
            "Malignant pleural mesothelioma — asbestos history",
            "Pleural effusion — fluid not thickening",
            "Extrapleural fat — no restriction on lung function"
        ],
        "lab_tests"       : [
            "Lung function tests — FVC and FEV1",
            "CT chest — characterise thickening",
            "Ultrasound guided biopsy if malignancy suspected",
            "Full blood count and ESR"
        ],
        "treatments"      : [
            "Monitoring with serial imaging if mild",
            "Pulmonary rehabilitation if symptomatic",
            "Oxygen therapy for resting hypoxia",
            "Decortication surgery if severe restriction"
        ],
        "prescription_note": (
            "Most cases require no medication. "
            "Surgical decortication if severe — "
            "specialist decision only."
        ),
        "follow_up"       : (
            "Annual chest X-ray and lung function tests. "
            "CT chest if any clinical change or new symptoms. "
            "Mesothelioma surveillance if asbestos exposure."
        ),
        "admission_criteria": (
            "Rarely requires admission. Admit only if "
            "respiratory failure from severe bilateral restriction."
        ),
        "prevention"      : [
            "Complete full course of TB treatment — DOTS",
            "Avoid asbestos strictly",
            "Treat pleural infections completely and promptly"
        ],
        "emergency_signs" : (
            "Seek urgent care if sudden significant "
            "worsening of breathlessness develops."
        )
    },

    "Hernia": {
        "full_name"       : "Diaphragmatic Hernia",
        "icd10"           : "K44.9",
        "description"     : (
            "Diaphragmatic hernia is herniation of abdominal "
            "contents through the diaphragm into the chest. "
            "Hiatus hernia is most common. Para-oesophageal "
            "hernia and strangulation are surgical emergencies."
        ),
        "imaging_findings": (
            "Air-fluid level or bowel gas in chest cavity. "
            "Elevated or distorted hemidiaphragm. "
            "Hiatus hernia shows retrocardiac gas shadow "
            "behind the heart on lateral view."
        ),
        "common_symptoms" : [
            "Chest pain or discomfort after meals",
            "Heartburn and acid reflux — GORD",
            "Regurgitation of food or acid",
            "Shortness of breath especially after eating",
            "Difficulty swallowing large boluses",
            "Nausea after meals",
            "Anaemia from chronic mucosal bleeding"
        ],
        "risk_factors"    : [
            "Obesity — increased intra-abdominal pressure",
            "Pregnancy — repeated increased pressure",
            "Chronic cough",
            "Chronic constipation and straining",
            "Age above 50",
            "Previous oesophageal or gastric surgery",
            "Trauma to diaphragm"
        ],
        "severity"        : {
            "mild"  : "Small sliding hiatus hernia — lifestyle modification",
            "mod"   : "Large hiatus hernia with significant GORD",
            "severe": "Para-oesophageal or strangulated — surgical emergency"
        },
        "urgency"         : {
            "low"   : (0.35, 0.65),
            "medium": (0.65, 0.80),
            "high"  : (0.80, 1.00)
        },
        "urgency_message" : {
            "low"   : "Surgical review within 2 weeks",
            "medium": "Surgical evaluation this week",
            "high"  : "Urgent surgical evaluation today"
        },
        "specialist"      : "General Surgeon / Thoracic Surgeon",
        "differential"    : [
            "Achalasia — dysphagia predominant, bird beak on barium",
            "Pulmonary abscess — cavitating lesion with air fluid",
            "Eventration of diaphragm — congenital thinning"
        ],
        "lab_tests"       : [
            "Barium swallow — defines anatomy and reflux",
            "Endoscopy — oesophagitis, Barrett's screening",
            "pH monitoring — confirms GORD",
            "CT chest and abdomen — surgical planning",
            "CBC — check for anaemia from bleeding"
        ],
        "treatments"      : [
            "Small frequent meals — avoid large portions",
            "Avoid lying flat for 3 hours after eating",
            "Weight loss if BMI above 25",
            "Proton pump inhibitors for acid suppression",
            "Laparoscopic Nissen fundoplication — definitive",
            "Emergency surgery for strangulation"
        ],
        "prescription_note": (
            "Omeprazole 20-40mg once daily before breakfast. "
            "Pantoprazole 40mg once daily as alternative. "
            "Surgical repair is definitive treatment for "
            "symptomatic or large hernia."
        ),
        "follow_up"       : (
            "Endoscopy at 6-12 months to monitor for Barrett's. "
            "Post-operative review at 6 weeks and 6 months."
        ),
        "admission_criteria": (
            "Para-oesophageal hernia with obstructive symptoms. "
            "Strangulated hernia — emergency surgical admission."
        ),
        "prevention"      : [
            "Maintain healthy weight — BMI below 25",
            "Quit smoking",
            "Avoid large meals and lying down after eating"
        ],
        "emergency_signs" : (
            "CALL EMERGENCY if sudden severe chest pain, "
            "vomiting blood, or complete inability to swallow."
        )
    }
}


# ══════════════════════════════════════════════════════
# OCCUPATION RISK DATABASE
# ══════════════════════════════════════════════════════

OCCUPATION_RISKS = {

    "coal_miner": {
        "keywords": ["mine", "miner", "mining", "coal", "colliery"],
        "suspected_diseases": ["Fibrosis", "Emphysema", "Pleural Thickening"],
        "warnings": [
            "Coal mining — high risk of Coal Workers Pneumoconiosis. Fine coal dust causes progressive lung scarring.",
            "Silicosis risk from silica dust in mines. Irreversible fibrosis. CT chest and spirometry essential.",
            "COPD significantly more common in coal miners even without smoking. Annual spirometry recommended.",
            "Any opacity must be evaluated in context of coal dust exposure before other diagnoses confirmed."
        ]
    },

    "silica_worker": {
        "keywords": ["quarry", "quarrying", "stone", "granite", "sandblasting", "ceramic", "pottery", "glass", "foundry", "silicon"],
        "suspected_diseases": ["Fibrosis", "Nodule", "Pleural Thickening"],
        "warnings": [
            "Silica dust exposure — high risk of silicosis. Irreversible progressive lung scarring.",
            "Eggshell calcification of lymph nodes is classic silicosis. CT chest essential.",
            "Silicosis also increases TB susceptibility significantly. TB screening is recommended."
        ]
    },

    "asbestos_worker": {
        "keywords": ["asbestos", "insulation", "shipyard", "ship building", "pipe fitting", "boiler", "demolition", "renovation", "old building"],
        "suspected_diseases": ["Pleural Thickening", "Fibrosis", "Mass"],
        "warnings": [
            "Asbestos exposure — risk of asbestosis, pleural plaques, and malignant mesothelioma. Latency 20-40 years.",
            "Any pleural thickening in asbestos-exposed worker must evaluate for mesothelioma urgently.",
            "Bilateral pleural plaques are pathognomonic of asbestos exposure. Annual CT surveillance."
        ]
    },

    "farmer": {
        "keywords": ["farmer", "farming", "agriculture", "crop", "field", "paddy", "wheat", "livestock", "poultry", "dairy", "mushroom", "compost"],
        "suspected_diseases": ["Infiltration", "Fibrosis", "Emphysema"],
        "warnings": [
            "Farming — risk of Farmer's Lung (hypersensitivity pneumonitis) from mouldy hay and organic dust.",
            "Poultry or bird exposure increases risk of Bird Fancier's Lung — interstitial lung disease.",
            "Pesticide and chemical inhalation can cause chemical pneumonitis.",
            "Organic dust toxic syndrome can mimic pneumonia. Exposure history essential for diagnosis."
        ]
    },

    "construction": {
        "keywords": ["construction", "builder", "cement", "concrete", "brick", "carpenter", "plumber", "roofer", "demolition", "contractor"],
        "suspected_diseases": ["Fibrosis", "Pleural Thickening", "Emphysema"],
        "warnings": [
            "Construction work — exposure to cement dust, silica, and potentially asbestos in older buildings.",
            "Demolition work on pre-1980 buildings carries significant asbestos risk.",
            "Cement dust causes chronic airway irritation, occupational asthma, and chronic bronchitis."
        ]
    },

    "healthcare": {
        "keywords": ["doctor", "nurse", "hospital", "healthcare", "ward", "icu", "medical", "nursing", "physiotherapist", "radiographer", "paramedic", "clinic", "lab"],
        "suspected_diseases": ["Infiltration", "Consolidation"],
        "warnings": [
            "Healthcare occupation — elevated TB exposure risk. Annual tuberculin test and chest X-ray recommended.",
            "Occupational exposure to airborne pathogens including drug-resistant TB and influenza.",
            "Any infiltrate in a healthcare worker must exclude TB before other diagnoses are pursued."
        ]
    },

    "welder": {
        "keywords": ["welder", "welding", "metal", "steel", "iron", "fabrication", "fitter", "lathe", "machinist", "metalwork"],
        "suspected_diseases": ["Fibrosis", "Infiltration", "Emphysema"],
        "warnings": [
            "Welding — risk of siderosis from iron oxide fumes. Diffuse fine nodularity on X-ray.",
            "Metal fume fever causes acute flu-like symptoms after heavy metal fume exposure.",
            "Manganese fume exposure in welding can cause pulmonary manganism and long term fibrosis.",
            "Mixed dust fibrosis from welding fumes combined with other dusts accelerates lung scarring."
        ]
    },

    "painter": {
        "keywords": ["painter", "paint", "coating", "spray", "varnish", "lacquer", "printing", "ink", "dye"],
        "suspected_diseases": ["Infiltration", "Emphysema"],
        "warnings": [
            "Painting — isocyanate exposure from spray paints causes occupational asthma.",
            "Solvent inhalation from oil-based paints causes chemical pneumonitis.",
            "Spray painting without protection significantly increases lung cancer risk."
        ]
    },

    "baker": {
        "keywords": ["baker", "bakery", "flour", "grain", "mill", "miller", "wheat", "bread", "biscuit", "confectionery"],
        "suspected_diseases": ["Infiltration", "Emphysema"],
        "warnings": [
            "Bakery — flour dust causes Baker's Asthma, one of the most common occupational asthmas.",
            "Grain dust causes chronic bronchitis and hypersensitivity pneumonitis from fungal contamination."
        ]
    },

    "textile": {
        "keywords": ["textile", "cotton", "fabric", "weaving", "spinning", "garment", "jute", "wool", "fibre"],
        "suspected_diseases": ["Emphysema", "Infiltration"],
        "warnings": [
            "Textile industry — byssinosis risk from cotton dust. Chest tightness on first day back after weekend.",
            "Jute and hemp dust causes similar respiratory symptoms. Chronic exposure leads to airflow limitation."
        ]
    }
}


# ══════════════════════════════════════════════════════
# MEDICAL CONDITIONS RISK DATABASE — EXPANDED
# ══════════════════════════════════════════════════════

CONDITION_RISKS = {

    "diabetes": {
        "keywords": ["diabetes", "diabetic", "dm", "sugar", "t2dm", "t1dm", "type 2", "type 1", "insulin", "metformin", "hba1c"],
        "suspected_diseases": ["Pneumonia", "Infiltration", "Consolidation"],
        "boost": {
            "Pneumonia": 0.07,
            "Infiltration": 0.05,
            "Consolidation": 0.06
        },
        "warnings": [
            "Diabetes mellitus — significantly increases susceptibility to bacterial pneumonia, TB, and Klebsiella.",
            "Diabetic patients often present with atypical infection symptoms. Fever may be absent even in severe pneumonia.",
            "Staphylococcal and Klebsiella pneumonia are disproportionately common in diabetics.",
            "Any lung infiltrate in a diabetic patient must exclude TB given the significantly elevated co-infection risk."
        ]
    },

    "hiv_aids": {
        "keywords": ["hiv", "aids", "immunocompromised", "immunosuppressed", "cd4", "art", "antiretroviral"],
        "suspected_diseases": ["Pneumonia", "Infiltration", "Consolidation", "Effusion"],
        "boost": {
            "Pneumonia": 0.09,
            "Infiltration": 0.08,
            "Consolidation": 0.07,
            "Effusion": 0.06
        },
        "warnings": [
            "HIV/AIDS — high risk of Pneumocystis jirovecii Pneumonia (PCP). Bilateral ground glass on X-ray.",
            "Disseminated TB is common in HIV. Miliary pattern — tiny nodules throughout both lungs.",
            "Kaposi Sarcoma can cause pulmonary infiltrates. Fungal infections including cryptococcosis possible.",
            "Bacterial pneumonia occurs more frequently and severely. Recurrent pneumonia is AIDS-defining."
        ]
    },

    "hypertension": {
        "keywords": ["hypertension", "htn", "high blood pressure", "high bp", "bp controlled", "antihypertensive"],
        "suspected_diseases": ["Cardiomegaly", "Edema"],
        "boost": {
            "Cardiomegaly": 0.08,
            "Edema": 0.06
        },
        "warnings": [
            "Hypertension — left ventricular hypertrophy and cardiomegaly are direct complications.",
            "Hypertensive crisis can cause acute pulmonary oedema even without pre-existing heart failure.",
            "Long standing uncontrolled hypertension leads to heart failure, pulmonary oedema, and effusions."
        ]
    },

    "heart_failure": {
        "keywords": ["heart failure", "cardiac failure", "ccf", "chf", "cardiomyopathy", "heart disease", "cardiac", "hfref", "hfpef"],
        "suspected_diseases": ["Edema", "Effusion", "Cardiomegaly"],
        "boost": {
            "Edema": 0.10,
            "Effusion": 0.09,
            "Cardiomegaly": 0.09
        },
        "warnings": [
            "Known heart failure — pulmonary oedema, cardiomegaly, and bilateral pleural effusions are expected.",
            "Acute decompensated heart failure causes rapid onset bilateral infiltrates. Any new breathlessness urgent.",
            "BNP or NT-proBNP level is essential to quantify decompensation severity."
        ]
    },

    "tuberculosis": {
        "keywords": ["tb", "tuberculosis", "koch", "mtb", "dots", "anti tb", "atd", "tb contact", "tb exposure"],
        "suspected_diseases": ["Infiltration", "Effusion", "Fibrosis", "Consolidation", "Pleural Thickening"],
        "boost": {
            "Infiltration": 0.09,
            "Effusion": 0.07,
            "Consolidation": 0.07,
            "Pleural Thickening": 0.06
        },
        "warnings": [
            "Known or past tuberculosis — reactivation TB must be actively considered. Upper lobe opacities are classic.",
            "Post-TB fibrosis and bronchiectasis are common sequelae causing recurrent infections and haemoptysis.",
            "Multidrug resistant TB (MDR-TB) must be excluded if patient was previously treated.",
            "TB can reactivate during immunosuppression, pregnancy, or malnutrition."
        ]
    },

    "copd": {
        "keywords": ["copd", "chronic obstructive", "emphysema", "chronic bronchitis", "obstructive airways", "airways disease"],
        "suspected_diseases": ["Emphysema", "Pneumothorax", "Infiltration"],
        "boost": {
            "Emphysema": 0.09,
            "Pneumothorax": 0.06,
            "Infiltration": 0.05
        },
        "warnings": [
            "Known COPD — acute exacerbation can be triggered by infection or pollution. Any new infiltrate = infective exacerbation.",
            "Pneumothorax risk elevated in COPD due to bullae. Sudden worsening must exclude pneumothorax urgently.",
            "Cor pulmonale and right heart failure are complications of severe COPD."
        ]
    },

    "cancer": {
        "keywords": ["cancer", "malignancy", "tumour", "tumor", "oncology", "chemotherapy", "radiotherapy", "radiation", "metastasis", "carcinoma", "lymphoma"],
        "suspected_diseases": ["Mass", "Nodule", "Effusion", "Infiltration"],
        "boost": {
            "Mass": 0.09,
            "Nodule": 0.07,
            "Effusion": 0.07,
            "Infiltration": 0.05
        },
        "warnings": [
            "Known malignancy — pulmonary metastases must be excluded as priority. Any new nodule needs urgent evaluation.",
            "Chemotherapy-induced pneumonitis can cause bilateral infiltrates mimicking infection.",
            "Immunosuppression from cancer treatment significantly increases opportunistic infection risk.",
            "Radiation pneumonitis can occur weeks to months after chest radiotherapy in radiation field."
        ]
    },

    "autoimmune": {
        "keywords": ["lupus", "sle", "rheumatoid", "ra", "autoimmune", "connective tissue", "sjogren", "scleroderma", "myositis", "vasculitis"],
        "suspected_diseases": ["Fibrosis", "Effusion", "Infiltration", "Pleural Thickening"],
        "boost": {
            "Fibrosis": 0.08,
            "Effusion": 0.07,
            "Infiltration": 0.05
        },
        "warnings": [
            "Autoimmune disease — interstitial lung disease (ILD) is a serious pulmonary complication.",
            "Lupus can cause pleuritis, pleural effusion, shrinking lung syndrome, and pulmonary haemorrhage.",
            "Rheumatoid ILD is progressive and can be severe. Early detection improves outcomes.",
            "Immunosuppressive therapy increases infection risk including opportunistic infections."
        ]
    },

    "renal": {
        "keywords": ["renal", "kidney", "ckd", "dialysis", "nephrotic", "nephritis", "kidney disease", "renal failure", "transplant"],
        "suspected_diseases": ["Edema", "Effusion", "Cardiomegaly"],
        "boost": {
            "Edema": 0.08,
            "Effusion": 0.07,
            "Cardiomegaly": 0.05
        },
        "warnings": [
            "Renal disease — fluid overload causes pulmonary oedema and pleural effusions.",
            "Dialysis patients at risk of dialysis-related effusions and pulmonary calcification.",
            "Nephrotic syndrome causes low oncotic pressure leading to bilateral effusions.",
            "Renal transplant recipients on immunosuppression at high risk of opportunistic pulmonary infections."
        ]
    },

    "asthma": {
        "keywords": ["asthma", "bronchial asthma", "reactive airway", "inhaler", "salbutamol", "montelukast"],
        "suspected_diseases": ["Emphysema", "Pneumothorax", "Atelectasis"],
        "boost": {
            "Atelectasis": 0.07,
            "Pneumothorax": 0.05
        },
        "warnings": [
            "Known asthma — severe attack can cause hyperinflation and mucus plugging with segmental atelectasis.",
            "Pneumothorax is a rare but serious complication of severe acute asthma. Sudden worsening needs urgent X-ray.",
            "ABPA is a complication of severe asthma causing central bronchiectasis and fleeting infiltrates."
        ]
    },

    "liver": {
        "keywords": ["liver", "cirrhosis", "hepatitis", "liver disease", "alcoholic liver", "fatty liver", "portal hypertension", "ascites"],
        "suspected_diseases": ["Effusion", "Edema"],
        "boost": {
            "Effusion": 0.09,
            "Edema": 0.06
        },
        "warnings": [
            "Liver cirrhosis — hepatic hydrothorax from ascitic fluid crossing diaphragm causes large right effusion.",
            "Low albumin from liver disease reduces oncotic pressure causing bilateral effusions and pulmonary oedema.",
            "Hepatopulmonary syndrome causes progressive hypoxia in liver disease. SpO2 monitoring important."
        ]
    },

    "obesity": {
        "keywords": ["obesity", "obese", "overweight", "bmi", "morbid obesity", "bariatric"],
        "suspected_diseases": ["Atelectasis", "Edema", "Cardiomegaly"],
        "boost": {
            "Atelectasis": 0.06,
            "Edema": 0.05,
            "Cardiomegaly": 0.05
        },
        "warnings": [
            "Obesity — basal atelectasis is common due to reduced diaphragm excursion and splinting.",
            "Obesity hypoventilation syndrome causes chronic hypoxia and pulmonary hypertension.",
            "Obese patients have higher risk of pulmonary embolism, aspiration pneumonia, and sleep apnoea."
        ]
    },

    "smoking": {
        "keywords": ["smoker", "smoking", "cigarette", "tobacco", "bidi", "hookah", "beedi", "pack year"],
        "suspected_diseases": ["Emphysema", "Mass", "Nodule", "Fibrosis"],
        "boost": {
            "Emphysema": 0.08,
            "Mass": 0.07,
            "Nodule": 0.06
        },
        "warnings": [
            "Smoking history — COPD, emphysema, and lung cancer risk significantly elevated.",
            "Any pulmonary mass or nodule in a smoker requires urgent malignancy evaluation.",
            "Even ex-smokers retain elevated lung cancer risk for 10-15 years after quitting."
        ]
    },

    "malnutrition": {
        "keywords": ["malnutrition", "malnourished", "underweight", "low weight", "poor nutrition", "vitamin deficiency"],
        "suspected_diseases": ["Infiltration", "Effusion", "Consolidation"],
        "boost": {
            "Infiltration": 0.07,
            "Consolidation": 0.06,
            "Effusion": 0.05
        },
        "warnings": [
            "Malnutrition — severely impairs immune function, increasing TB, pneumonia, and fungal infection risk.",
            "Low albumin from malnutrition causes pleural effusion and pulmonary oedema without cardiac cause.",
            "TB and malnutrition are strongly linked in India. Weight loss with any lung opacity = TB until excluded."
        ]
    },

    "alcohol": {
        "keywords": ["alcohol", "alcoholic", "alcoholism", "alcohol dependence", "drinks daily", "heavy drinker"],
        "suspected_diseases": ["Pneumonia", "Effusion", "Consolidation"],
        "boost": {
            "Pneumonia": 0.08,
            "Consolidation": 0.07,
            "Effusion": 0.06
        },
        "warnings": [
            "Alcohol use — aspiration pneumonia risk significantly elevated due to impaired cough reflex.",
            "Klebsiella pneumonia is particularly associated with alcohol dependence — causes upper lobe destruction.",
            "Alcoholic liver disease causes hepatic hydrothorax and pleural effusion.",
            "Immune suppression from alcohol increases TB susceptibility significantly."
        ]
    },

    "pregnancy": {
        "keywords": ["pregnant", "pregnancy", "antenatal", "postnatal", "postpartum", "obstetric", "maternal"],
        "suspected_diseases": ["Edema", "Effusion", "Pneumonia"],
        "boost": {
            "Edema": 0.06,
            "Pneumonia": 0.05
        },
        "warnings": [
            "Pregnancy — peripartum cardiomyopathy can cause acute pulmonary oedema in late pregnancy or postpartum.",
            "Pre-eclampsia with pulmonary oedema is a medical emergency requiring immediate delivery.",
            "Pneumonia in pregnancy is more severe and requires hospital admission in most cases."
        ]
    }
}


# ══════════════════════════════════════════════════════
# MEDICATION RISK DATABASE — EXPANDED
# ══════════════════════════════════════════════════════

MEDICATION_RISKS = {

    "amiodarone": {
        "keywords": ["amiodarone", "cordarone"],
        "suspected_diseases": ["Fibrosis", "Infiltration"],
        "boost": {"Fibrosis": 0.08, "Infiltration": 0.07},
        "warnings": [
            "Amiodarone — pulmonary toxicity in 5-15% of patients. Causes pneumonitis or fibrosis.",
            "Amiodarone lung toxicity can be life threatening and mimics infection on X-ray.",
            "Annual chest X-ray and pulmonary function tests mandatory for all on long term amiodarone."
        ]
    },

    "methotrexate": {
        "keywords": ["methotrexate", "mtx", "rheumatrex"],
        "suspected_diseases": ["Infiltration", "Fibrosis"],
        "boost": {"Infiltration": 0.07, "Fibrosis": 0.06},
        "warnings": [
            "Methotrexate — drug-induced pneumonitis in 3-5% of patients. Acute breathlessness and bilateral infiltrates.",
            "Methotrexate lung toxicity can be severe and life threatening. Drug must be stopped immediately if suspected.",
            "Baseline chest X-ray before starting and annual monitoring are standard of care."
        ]
    },

    "steroids": {
        "keywords": ["prednisolone", "prednisone", "dexamethasone", "methylprednisolone", "hydrocortisone", "steroid", "corticosteroid", "budesonide"],
        "suspected_diseases": ["Infiltration", "Consolidation", "Pneumonia"],
        "boost": {"Pneumonia": 0.07, "Infiltration": 0.06, "Consolidation": 0.06},
        "warnings": [
            "Systemic steroids — immunosuppression increases risk of PCP, fungal pneumonia, and atypical TB.",
            "Steroid-induced diabetes further increases infection susceptibility.",
            "Infections in steroid users may present without fever. High index of suspicion needed."
        ]
    },

    "nitrofurantoin": {
        "keywords": ["nitrofurantoin", "macrobid", "macrodantin"],
        "suspected_diseases": ["Fibrosis", "Infiltration"],
        "boost": {"Fibrosis": 0.07, "Infiltration": 0.06},
        "warnings": [
            "Nitrofurantoin — acute hypersensitivity or chronic progressive fibrosis.",
            "Acute toxicity: sudden breathlessness, fever, bilateral infiltrates. Responds to stopping drug.",
            "Chronic toxicity: progressive fibrosis that may not resolve even after stopping."
        ]
    },

    "immunosuppressants": {
        "keywords": ["tacrolimus", "cyclosporine", "mycophenolate", "azathioprine", "sirolimus", "everolimus", "immunosuppressant"],
        "suspected_diseases": ["Infiltration", "Pneumonia", "Consolidation"],
        "boost": {"Pneumonia": 0.08, "Infiltration": 0.07, "Consolidation": 0.06},
        "warnings": [
            "Immunosuppressants — high risk of opportunistic infections including PCP, CMV, and invasive fungal.",
            "Any new infiltrate in transplant recipient must be investigated urgently with BAL.",
            "Drug-induced ILD is also possible. Clinical correlation and specialist input essential."
        ]
    },

    "immunotherapy": {
        "keywords": ["pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "ipilimumab", "checkpoint", "immunotherapy", "pd-1", "pd-l1", "ctla-4"],
        "suspected_diseases": ["Infiltration", "Fibrosis", "Consolidation"],
        "boost": {"Infiltration": 0.08, "Fibrosis": 0.06},
        "warnings": [
            "Checkpoint inhibitor immunotherapy — immune related pneumonitis in 5-10% of patients.",
            "Immunotherapy pneumonitis can appear as organising pneumonia or diffuse alveolar damage.",
            "Any new respiratory symptom in a patient on checkpoint inhibitors must be evaluated urgently.",
            "High dose steroids are first line treatment for immune related pneumonitis."
        ]
    },

    "chemotherapy": {
        "keywords": ["cyclophosphamide", "bleomycin", "busulfan", "carmustine", "gemcitabine", "taxol", "paclitaxel", "docetaxel", "chemotherapy", "chemo"],
        "suspected_diseases": ["Fibrosis", "Infiltration"],
        "boost": {"Fibrosis": 0.08, "Infiltration": 0.06},
        "warnings": [
            "Chemotherapy — bleomycin, busulfan, cyclophosphamide are causes of pulmonary fibrosis and pneumonitis.",
            "Chemotherapy-induced lung toxicity must be distinguished from infection and disease progression.",
            "Bleomycin toxicity risk increases significantly above 400 units total lifetime dose."
        ]
    },

    "ace_inhibitor": {
        "keywords": ["enalapril", "lisinopril", "ramipril", "perindopril", "captopril", "ace inhibitor", "acei"],
        "suspected_diseases": [],
        "boost": {},
        "warnings": [
            "ACE inhibitor use — dry persistent cough in up to 20% of patients. X-ray is normal.",
            "If cough is main symptom and X-ray clear, ACE inhibitor cough should be considered."
        ]
    },

    "antibiotic": {
        "keywords": ["amoxicillin", "azithromycin", "doxycycline", "ciprofloxacin", "levofloxacin", "ceftriaxone", "antibiotic", "antibiotics"],
        "suspected_diseases": [],
        "boost": {},
        "warnings": [
            "Currently on antibiotics — X-ray findings may lag clinical improvement by 4-6 weeks."
        ]
    },

    "antifungal": {
        "keywords": ["voriconazole", "itraconazole", "fluconazole", "amphotericin", "antifungal", "caspofungin"],
        "suspected_diseases": ["Infiltration", "Consolidation"],
        "boost": {"Infiltration": 0.05},
        "warnings": [
            "Antifungal therapy — suggests known or suspected fungal pulmonary infection.",
            "Voriconazole-induced photosensitivity rarely causes pulmonary toxicity. Monitor closely.",
            "Invasive aspergillosis or mucormycosis must be excluded if antifungal started for pulmonary indication."
        ]
    },

    "anticoagulant": {
        "keywords": ["warfarin", "heparin", "rivaroxaban", "apixaban", "dabigatran", "anticoagulant", "blood thinner", "clexane", "enoxaparin"],
        "suspected_diseases": ["Effusion", "Infiltration"],
        "boost": {"Effusion": 0.05},
        "warnings": [
            "Anticoagulant use — haemothorax risk if any chest trauma or procedure.",
            "Anticoagulation for PE may indicate underlying hypercoagulable state — monitor for recurrence.",
            "Pulmonary haemorrhage is a rare but serious complication of anticoagulation."
        ]
    },

    "anti_tb": {
        "keywords": ["rifampicin", "isoniazid", "pyrazinamide", "ethambutol", "dots", "anti tubercular", "atd", "rh", "hrze"],
        "suspected_diseases": ["Infiltration", "Fibrosis", "Pleural Thickening"],
        "boost": {"Infiltration": 0.07, "Fibrosis": 0.05},
        "warnings": [
            "Anti-TB treatment — patient has active or recent TB. Lung opacity must be evaluated in this context.",
            "Drug-induced liver injury from anti-TB drugs can rarely cause pulmonary hypersensitivity.",
            "Paradoxical worsening of X-ray findings can occur in first weeks of TB treatment — do not stop treatment."
        ]
    },

    "diuretic": {
        "keywords": ["furosemide", "lasix", "torsemide", "spironolactone", "hydrochlorothiazide", "diuretic", "water tablet"],
        "suspected_diseases": ["Edema", "Effusion", "Cardiomegaly"],
        "boost": {"Edema": 0.06, "Effusion": 0.05, "Cardiomegaly": 0.05},
        "warnings": [
            "Diuretic use — suggests known cardiac failure, hypertension, or renal disease causing fluid retention.",
            "Bilateral infiltrates despite diuretic therapy may represent diuretic-resistant cardiac failure.",
            "Hyponatraemia from diuretics can cause confusion — important to check electrolytes."
        ]
    },

    "beta_blocker": {
        "keywords": ["metoprolol", "atenolol", "bisoprolol", "carvedilol", "propranolol", "beta blocker", "nebivolol"],
        "suspected_diseases": ["Cardiomegaly", "Edema"],
        "boost": {"Cardiomegaly": 0.05},
        "warnings": [
            "Beta blocker use — suggests underlying cardiac disease, heart failure, or hypertension.",
            "Rarely beta blockers can cause pulmonary hypersensitivity reaction with bilateral infiltrates."
        ]
    }
}


# ══════════════════════════════════════════════════════
# SYMPTOM CORRELATION DATABASE — EXPANDED
# Multi-symptom combinations give stronger, more accurate boosts
# ══════════════════════════════════════════════════════

SYMPTOM_CORRELATIONS = {

    "haemoptysis": {
        "suspected_diseases": ["Mass", "Nodule", "Infiltration"],
        "boost": {"Mass": 0.09, "Nodule": 0.07, "Infiltration": 0.06},
        "warning": (
            "Haemoptysis (coughing blood) is a RED FLAG symptom. "
            "Urgent investigation mandatory to exclude lung cancer and TB."
        )
    },

    "weight_loss_night_sweats": {
        "condition": lambda s: s.get("weight_loss") and s.get("night_sweats"),
        "suspected_diseases": ["Infiltration", "Mass", "Consolidation"],
        "boost": {"Infiltration": 0.09, "Mass": 0.07, "Consolidation": 0.06},
        "warning": (
            "Weight loss combined with night sweats — classic B-symptom pattern. "
            "Strong concern for tuberculosis or lymphoma. Urgent TB workup recommended."
        )
    },

    "fever_productive_cough": {
        "condition": lambda s: s.get("fever") and s.get("sputum"),
        "suspected_diseases": ["Pneumonia", "Consolidation", "Infiltration"],
        "boost": {"Pneumonia": 0.08, "Consolidation": 0.07, "Infiltration": 0.05},
        "warning": (
            "Fever with productive cough — bacterial pneumonia most likely. "
            "Sputum culture before antibiotics is essential."
        )
    },

    "fever_weight_loss": {
        "condition": lambda s: s.get("fever") and s.get("weight_loss"),
        "suspected_diseases": ["Infiltration", "Consolidation", "Effusion"],
        "boost": {"Infiltration": 0.08, "Consolidation": 0.07},
        "warning": (
            "Fever with significant weight loss — tuberculosis must be "
            "actively excluded before any other diagnosis. Sputum AFB essential."
        )
    },

    "breathless_orthopnoea": {
        "condition": lambda s: s.get("breathless") and s.get("orthopnoea"),
        "suspected_diseases": ["Edema", "Effusion", "Cardiomegaly"],
        "boost": {"Edema": 0.09, "Effusion": 0.07, "Cardiomegaly": 0.06},
        "warning": (
            "Breathlessness with inability to lie flat (orthopnoea) — "
            "cardiac failure until proven otherwise. BNP and echocardiogram urgently."
        )
    },

    "orthopnoea_swelling": {
        "condition": lambda s: s.get("orthopnoea") and s.get("swelling"),
        "suspected_diseases": ["Edema", "Effusion", "Cardiomegaly"],
        "boost": {"Edema": 0.10, "Effusion": 0.08, "Cardiomegaly": 0.07},
        "warning": (
            "Orthopnoea combined with leg swelling — strongly suggests cardiac failure. "
            "BNP and echocardiogram are priority investigations."
        )
    },

    "breathless_swelling_fatigue": {
        "condition": lambda s: s.get("breathless") and s.get("swelling") and s.get("fatigue"),
        "suspected_diseases": ["Edema", "Cardiomegaly", "Effusion"],
        "boost": {"Edema": 0.09, "Cardiomegaly": 0.08, "Effusion": 0.07},
        "warning": (
            "Breathlessness with leg swelling and fatigue — classic cardiac failure triad. "
            "Urgent cardiac evaluation required."
        )
    },

    "tb_contact": {
        "suspected_diseases": ["Infiltration", "Consolidation", "Effusion"],
        "boost": {"Infiltration": 0.09, "Consolidation": 0.07, "Effusion": 0.06},
        "warning": (
            "TB contact history — active tuberculosis must be excluded before any other diagnosis. "
            "Mantoux or IGRA and sputum AFB smear are essential first steps."
        )
    },

    "night_sweats_alone": {
        "condition": lambda s: s.get("night_sweats") and not s.get("weight_loss"),
        "suspected_diseases": ["Infiltration", "Effusion"],
        "boost": {"Infiltration": 0.06},
        "warning": (
            "Night sweats — consider TB, lymphoma, or fungal infection. "
            "Clinical correlation and further investigation needed."
        )
    },

    "pleuritic_pain_breathless": {
        "condition": lambda s: s.get("pleuritic_pain") and s.get("breathless"),
        "suspected_diseases": ["Pneumothorax", "Effusion", "Consolidation"],
        "boost": {"Pneumothorax": 0.07, "Effusion": 0.06, "Consolidation": 0.05},
        "warning": (
            "Pleuritic pain with breathlessness — pneumothorax and pleural effusion "
            "must be excluded urgently. Immediate chest X-ray essential."
        )
    },

    "sudden_breathless_chest_pain": {
        "condition": lambda s: s.get("breathless") and s.get("chest_pain") and not s.get("fever"),
        "suspected_diseases": ["Pneumothorax", "Effusion"],
        "boost": {"Pneumothorax": 0.08, "Effusion": 0.05},
        "warning": (
            "Sudden breathlessness with chest pain and no fever — "
            "pneumothorax and pulmonary embolism must be excluded urgently."
        )
    },

    "haemoptysis_weight_loss": {
        "condition": lambda s: s.get("haemoptysis") and s.get("weight_loss"),
        "suspected_diseases": ["Mass", "Infiltration", "Consolidation"],
        "boost": {"Mass": 0.10, "Infiltration": 0.08},
        "warning": (
            "Haemoptysis combined with weight loss — extremely high concern for "
            "lung cancer or TB. Urgent CT chest and bronchoscopy required. Do not delay."
        )
    },

    "cyanosis": {
        "suspected_diseases": [],
        "boost": {},
        "warning": (
            "Cyanosis present — severe hypoxia. SpO2 monitoring and ABG required immediately. "
            "High flow oxygen should be administered without delay."
        )
    },

    "dust_exposure": {
        "suspected_diseases": ["Fibrosis", "Emphysema", "Infiltration"],
        "boost": {"Fibrosis": 0.07, "Emphysema": 0.05},
        "warning": (
            "Dust or chemical exposure — occupational lung disease must be considered. "
            "Full occupational history including duration and type of exposure is essential."
        )
    },

    "recent_surgery": {
        "suspected_diseases": ["Atelectasis", "Effusion"],
        "boost": {"Atelectasis": 0.09, "Effusion": 0.06},
        "warning": (
            "Recent surgery or prolonged bed rest — post-operative atelectasis and "
            "pulmonary embolism are primary concerns."
        )
    },

    "family_lung": {
        "suspected_diseases": ["Fibrosis", "Mass"],
        "boost": {"Fibrosis": 0.06, "Mass": 0.05},
        "warning": (
            "Family history of lung disease — genetic predisposition to pulmonary "
            "fibrosis (TERT/TERC mutations) and lung cancer increases personal risk."
        )
    },

    "chronic_cough_no_fever": {
        "condition": lambda s: s.get("dry_cough") and not s.get("fever") and not s.get("sputum"),
        "suspected_diseases": ["Fibrosis", "Mass", "Nodule"],
        "boost": {"Fibrosis": 0.06, "Mass": 0.05},
        "warning": (
            "Chronic dry cough without fever or sputum — consider pulmonary fibrosis, "
            "malignancy, or ACE inhibitor cough. Spirometry and CT chest may be needed."
        )
    },

    "palpitations_breathless": {
        "condition": lambda s: s.get("palpitations") and s.get("breathless"),
        "suspected_diseases": ["Cardiomegaly", "Edema", "Effusion"],
        "boost": {"Cardiomegaly": 0.07, "Edema": 0.06},
        "warning": (
            "Palpitations with breathlessness — cardiac arrhythmia causing decompensated "
            "heart failure. ECG and echocardiogram urgently required."
        )
    },

    "loss_appetite_fatigue_cough": {
        "condition": lambda s: s.get("loss_appetite") and s.get("fatigue") and (s.get("cough") or s.get("dry_cough")),
        "suspected_diseases": ["Infiltration", "Mass", "Consolidation"],
        "boost": {"Infiltration": 0.07, "Mass": 0.06},
        "warning": (
            "Loss of appetite with fatigue and cough — consider TB, malignancy, or "
            "chronic infection. Duration of symptoms is key to narrowing the diagnosis."
        )
    }
}


# ══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════

def get_urgency_level(disease, probability):
    """Returns urgency level based on probability"""
    if disease not in DISEASE_INFO:
        return "low"
    levels = DISEASE_INFO[disease]["urgency"]
    if probability >= levels["high"][0]:
        return "high"
    elif probability >= levels["medium"][0]:
        return "medium"
    else:
        return "low"


def get_disease_report(disease, probability):
    """Returns complete clinical report for a disease"""
    if disease not in DISEASE_INFO:
        return None
    info    = DISEASE_INFO[disease]
    urgency = get_urgency_level(disease, probability)
    return {
        "disease"           : disease,
        "full_name"         : info["full_name"],
        "icd10"             : info.get("icd10", ""),
        "probability"       : round(probability * 100, 1),
        "urgency"           : urgency,
        "urgency_message"   : info["urgency_message"][urgency],
        "description"       : info["description"],
        "imaging_findings"  : info.get("imaging_findings", ""),
        "symptoms"          : info["common_symptoms"],
        "risk_factors"      : info.get("risk_factors", []),
        "severity"          : info.get("severity", {}),
        "specialist"        : info["specialist"],
        "differential"      : info.get("differential", []),
        "lab_tests"         : info.get("lab_tests", []),
        "treatments"        : info["treatments"],
        "prescription_note" : info["prescription_note"],
        "follow_up"         : info.get("follow_up", ""),
        "admission_criteria": info.get("admission_criteria", ""),
        "prevention"        : info["prevention"],
        "emergency_signs"   : info["emergency_signs"]
    }


def get_risk_summary(patient_age, patient_gender,
                     smoking, symptoms,
                     occupation="",
                     conditions="",
                     medications=""):
    """
    Returns:
      notes           — list of warning strings
      extra_diseases  — list of additionally suspected disease names
    """
    notes          = []
    extra_diseases = set()
    disease_boosts = {}  # accumulate boosts per disease

    def add_boost(disease, amount):
        disease_boosts[disease] = min(
            disease_boosts.get(disease, 0.0) + amount,
            0.10  # hard cap at 10%
        )

    # ── Age risks ─────────────────────────────────────
    try:
        age = int(patient_age)
        if age > 65:
            notes.append(
                "Age above 65 increases risk of pneumonia, "
                "cardiac disease, and malignancy significantly."
            )
            add_boost("Pneumonia", 0.04)
            add_boost("Cardiomegaly", 0.03)
        if age < 40:
            notes.append(
                "Younger age makes primary spontaneous "
                "pneumothorax and atypical infections more likely."
            )
            add_boost("Pneumothorax", 0.04)
    except:
        pass

    # ── Smoking risks ─────────────────────────────────
    if smoking in ["yes", "past"]:
        notes.append(
            "Smoking history significantly increases risk "
            "of COPD, emphysema, lung cancer, and cardiovascular disease."
        )
        extra_diseases.update(["Emphysema", "Mass", "Nodule"])
        add_boost("Emphysema", 0.07)
        add_boost("Mass", 0.06)
        add_boost("Nodule", 0.05)

    # ── Symptom combination risks — MULTI-SYMPTOM FIRST ──
    # Multi-symptom combinations are evaluated before single
    # symptoms to give stronger boosts for clear clinical patterns

    # Combination: orthopnoea + swelling (strongest cardiac signal)
    if symptoms.get("orthopnoea") and symptoms.get("swelling"):
        sc = SYMPTOM_CORRELATIONS["orthopnoea_swelling"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    # Combination: breathless + swelling + fatigue
    elif symptoms.get("breathless") and symptoms.get("swelling") and symptoms.get("fatigue"):
        sc = SYMPTOM_CORRELATIONS["breathless_swelling_fatigue"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    # Combination: breathless + orthopnoea
    elif symptoms.get("breathless") and symptoms.get("orthopnoea"):
        sc = SYMPTOM_CORRELATIONS["breathless_orthopnoea"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    # Combination: haemoptysis + weight loss (strongest malignancy/TB signal)
    if symptoms.get("haemoptysis") and symptoms.get("weight_loss"):
        sc = SYMPTOM_CORRELATIONS["haemoptysis_weight_loss"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    # Combination: weight loss + night sweats (TB/lymphoma signal)
    elif symptoms.get("weight_loss") and symptoms.get("night_sweats"):
        sc = SYMPTOM_CORRELATIONS["weight_loss_night_sweats"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    # Combination: fever + weight loss
    elif symptoms.get("fever") and symptoms.get("weight_loss"):
        sc = SYMPTOM_CORRELATIONS["fever_weight_loss"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    # Combination: fever + productive cough
    if symptoms.get("fever") and symptoms.get("sputum"):
        sc = SYMPTOM_CORRELATIONS["fever_productive_cough"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    # Combination: pleuritic pain + breathless
    if symptoms.get("pleuritic_pain") and symptoms.get("breathless"):
        sc = SYMPTOM_CORRELATIONS["pleuritic_pain_breathless"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    # Combination: sudden breathless + chest pain, no fever
    if symptoms.get("breathless") and symptoms.get("chest_pain") and not symptoms.get("fever"):
        sc = SYMPTOM_CORRELATIONS["sudden_breathless_chest_pain"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    # Combination: palpitations + breathless
    if symptoms.get("palpitations") and symptoms.get("breathless"):
        sc = SYMPTOM_CORRELATIONS["palpitations_breathless"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    # Combination: loss of appetite + fatigue + cough
    if symptoms.get("loss_appetite") and symptoms.get("fatigue") and (symptoms.get("cough") or symptoms.get("dry_cough")):
        sc = SYMPTOM_CORRELATIONS["loss_appetite_fatigue_cough"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    # Combination: chronic dry cough, no fever, no sputum
    if symptoms.get("dry_cough") and not symptoms.get("fever") and not symptoms.get("sputum"):
        sc = SYMPTOM_CORRELATIONS["chronic_cough_no_fever"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    # ── Single symptom risks ──────────────────────────
    if symptoms.get("haemoptysis") and not symptoms.get("weight_loss"):
        sc = SYMPTOM_CORRELATIONS["haemoptysis"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    if symptoms.get("tb_contact"):
        sc = SYMPTOM_CORRELATIONS["tb_contact"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    if symptoms.get("night_sweats") and not symptoms.get("weight_loss"):
        sc = SYMPTOM_CORRELATIONS["night_sweats_alone"]
        notes.append(sc["warning"])
        extra_diseases.update(sc.get("suspected_diseases", []))
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    if symptoms.get("cyanosis"):
        sc = SYMPTOM_CORRELATIONS["cyanosis"]
        notes.append(sc["warning"])

    if symptoms.get("dust_exposure"):
        sc = SYMPTOM_CORRELATIONS["dust_exposure"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    if symptoms.get("recent_surgery"):
        sc = SYMPTOM_CORRELATIONS["recent_surgery"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    if symptoms.get("family_lung"):
        sc = SYMPTOM_CORRELATIONS["family_lung"]
        notes.append(sc["warning"])
        extra_diseases.update(sc["suspected_diseases"])
        for d, b in sc.get("boost", {}).items():
            add_boost(d, b)

    # ── Occupation risks ──────────────────────────────
    if occupation:
        occ = occupation.lower()
        for risk_key, risk_data in OCCUPATION_RISKS.items():
            if any(kw in occ for kw in risk_data["keywords"]):
                for warning in risk_data["warnings"]:
                    notes.append(warning)
                extra_diseases.update(risk_data["suspected_diseases"])

    # ── Medical conditions risks ───────────────────────
    if conditions:
        cond = conditions.lower()
        for risk_key, risk_data in CONDITION_RISKS.items():
            if any(kw in cond for kw in risk_data["keywords"]):
                for warning in risk_data["warnings"]:
                    notes.append(warning)
                extra_diseases.update(risk_data["suspected_diseases"])
                for d, b in risk_data.get("boost", {}).items():
                    add_boost(d, b)

    # ── Medication risks ──────────────────────────────
    if medications:
        meds = medications.lower()
        for risk_key, risk_data in MEDICATION_RISKS.items():
            if any(kw in meds for kw in risk_data["keywords"]):
                for warning in risk_data["warnings"]:
                    notes.append(warning)
                extra_diseases.update(risk_data["suspected_diseases"])
                for d, b in risk_data.get("boost", {}).items():
                    add_boost(d, b)

    return notes, list(extra_diseases)


def get_disease_boosts(patient_age, patient_gender,
                       smoking, symptoms,
                       occupation="",
                       conditions="",
                       medications=""):
    """
    Returns a dict of disease_name → boost_amount (0.0 to 0.10)
    Used by app.py to apply history boosts to AI predictions.
    All boosts capped at 0.10 (10%) — image AI always contributes >=80%.
    """
    _, _ = get_risk_summary(
        patient_age, patient_gender, smoking, symptoms,
        occupation, conditions, medications
    )

    # Re-run boost accumulation
    disease_boosts = {}

    def add_boost(disease, amount):
        disease_boosts[disease] = min(
            disease_boosts.get(disease, 0.0) + amount,
            0.10
        )

    try:
        age = int(patient_age)
        if age > 65:
            add_boost("Pneumonia", 0.04)
            add_boost("Cardiomegaly", 0.03)
        if age < 40:
            add_boost("Pneumothorax", 0.04)
    except:
        pass

    if smoking in ["yes", "past"]:
        add_boost("Emphysema", 0.07)
        add_boost("Mass", 0.06)
        add_boost("Nodule", 0.05)

    # Symptom combinations
    sym = symptoms or {}

    if sym.get("orthopnoea") and sym.get("swelling"):
        for d, b in SYMPTOM_CORRELATIONS["orthopnoea_swelling"].get("boost", {}).items():
            add_boost(d, b)
    elif sym.get("breathless") and sym.get("swelling") and sym.get("fatigue"):
        for d, b in SYMPTOM_CORRELATIONS["breathless_swelling_fatigue"].get("boost", {}).items():
            add_boost(d, b)
    elif sym.get("breathless") and sym.get("orthopnoea"):
        for d, b in SYMPTOM_CORRELATIONS["breathless_orthopnoea"].get("boost", {}).items():
            add_boost(d, b)

    if sym.get("haemoptysis") and sym.get("weight_loss"):
        for d, b in SYMPTOM_CORRELATIONS["haemoptysis_weight_loss"].get("boost", {}).items():
            add_boost(d, b)
    elif sym.get("weight_loss") and sym.get("night_sweats"):
        for d, b in SYMPTOM_CORRELATIONS["weight_loss_night_sweats"].get("boost", {}).items():
            add_boost(d, b)
    elif sym.get("fever") and sym.get("weight_loss"):
        for d, b in SYMPTOM_CORRELATIONS["fever_weight_loss"].get("boost", {}).items():
            add_boost(d, b)

    if sym.get("fever") and sym.get("sputum"):
        for d, b in SYMPTOM_CORRELATIONS["fever_productive_cough"].get("boost", {}).items():
            add_boost(d, b)

    if sym.get("pleuritic_pain") and sym.get("breathless"):
        for d, b in SYMPTOM_CORRELATIONS["pleuritic_pain_breathless"].get("boost", {}).items():
            add_boost(d, b)

    if sym.get("breathless") and sym.get("chest_pain") and not sym.get("fever"):
        for d, b in SYMPTOM_CORRELATIONS["sudden_breathless_chest_pain"].get("boost", {}).items():
            add_boost(d, b)

    if sym.get("palpitations") and sym.get("breathless"):
        for d, b in SYMPTOM_CORRELATIONS["palpitations_breathless"].get("boost", {}).items():
            add_boost(d, b)

    if sym.get("loss_appetite") and sym.get("fatigue") and (sym.get("cough") or sym.get("dry_cough")):
        for d, b in SYMPTOM_CORRELATIONS["loss_appetite_fatigue_cough"].get("boost", {}).items():
            add_boost(d, b)

    if sym.get("dry_cough") and not sym.get("fever") and not sym.get("sputum"):
        for d, b in SYMPTOM_CORRELATIONS["chronic_cough_no_fever"].get("boost", {}).items():
            add_boost(d, b)

    if sym.get("haemoptysis") and not sym.get("weight_loss"):
        for d, b in SYMPTOM_CORRELATIONS["haemoptysis"].get("boost", {}).items():
            add_boost(d, b)

    if sym.get("tb_contact"):
        for d, b in SYMPTOM_CORRELATIONS["tb_contact"].get("boost", {}).items():
            add_boost(d, b)

    if sym.get("night_sweats") and not sym.get("weight_loss"):
        for d, b in SYMPTOM_CORRELATIONS["night_sweats_alone"].get("boost", {}).items():
            add_boost(d, b)

    if sym.get("dust_exposure"):
        for d, b in SYMPTOM_CORRELATIONS["dust_exposure"].get("boost", {}).items():
            add_boost(d, b)

    if sym.get("recent_surgery"):
        for d, b in SYMPTOM_CORRELATIONS["recent_surgery"].get("boost", {}).items():
            add_boost(d, b)

    if sym.get("family_lung"):
        for d, b in SYMPTOM_CORRELATIONS["family_lung"].get("boost", {}).items():
            add_boost(d, b)

    # Conditions
    if conditions:
        cond = conditions.lower()
        for risk_key, risk_data in CONDITION_RISKS.items():
            if any(kw in cond for kw in risk_data["keywords"]):
                for d, b in risk_data.get("boost", {}).items():
                    add_boost(d, b)

    # Medications
    if medications:
        meds = medications.lower()
        for risk_key, risk_data in MEDICATION_RISKS.items():
            if any(kw in meds for kw in risk_data["keywords"]):
                for d, b in risk_data.get("boost", {}).items():
                    add_boost(d, b)

    return disease_boosts


def get_clinical_summary(occupation="", conditions="",
                         medications="", symptoms=None):
    """
    Returns a plain text clinical context summary for use in reports
    """
    if symptoms is None:
        symptoms = {}

    notes, extra = get_risk_summary(
        "", "", "", symptoms,
        occupation, conditions, medications
    )

    lines = []
    if notes:
        lines.append("CLINICAL CONTEXT FROM PATIENT HISTORY:")
        for note in notes:
            lines.append(f"  • {note}")
    if extra:
        lines.append("\nADDITIONALLY SUSPECTED FROM HISTORY:")
        for d in extra:
            lines.append(f"  • {d}")
    return "\n".join(lines)