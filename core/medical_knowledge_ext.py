# medical_knowledge_ext.py
# AyShCXR — canonical knowledge-base extension. 2026-08-10
# by Subhrakant Sethi & Ayush Singh
#
# medical_knowledge.py covers the NIH-14 findings and is left completely
# untouched. This module appends the 7 findings that CheXpert models emit but
# NIH never had, taking the knowledge base to the 21 canonical findings defined
# in core/disease_ontology.py.
#
# Two of the seven are NOT pathologies:
#   Support Devices — a tube / line / pacemaker is visible. Useful context, but
#                     there is nothing to treat and no urgency to assign.
#   No Finding      — the study looks normal. Never triaged, never treated.
# disease_ontology tags these kind="device" / kind="normal"; symptom_fusion
# already skips them; is_pathology() below lets app.py do the same.
#
# Clinical content is deliberately conservative. These are broad radiological
# descriptors, not specific diagnoses, so each one routes toward correlation and
# referral rather than naming a treatment. Every entry carries a real guideline
# citation in EXTENDED_CLINICAL_SOURCES.

EXTENDED_DISEASE_INFO = {

    "Enlarged Cardiomediastinum": {
        "full_name": "Enlarged Cardiomediastinum",
        "icd10": "R93.1",
        "description": "Widening of the cardiac and mediastinal silhouette. This is a radiological observation, not a diagnosis - it may reflect an enlarged heart, mediastinal fluid or mass, vascular abnormality, or simply a projection artefact from an AP or rotated film.",
        "imaging_findings": "Mediastinal width exceeding 8 cm on PA film, or cardiothoracic ratio above 0.5 with indistinct mediastinal borders. Always check film projection first - AP and supine films magnify the mediastinum.",
        "common_symptoms": [
            "Often none - frequently an incidental finding",
            "Shortness of breath",
            "Chest discomfort or pressure",
            "Hoarseness of voice",
            "Difficulty swallowing",
            "Facial or arm swelling",
        ],
        "risk_factors": [
            "Hypertension",
            "Existing heart failure",
            "Aortic aneurysm or dissection",
            "Mediastinal lymphadenopathy - infection or malignancy",
            "Retrosternal thyroid enlargement",
            "Recent chest trauma",
        ],
        "severity": {
            "mild": "Borderline widening - repeat PA film to confirm",
            "mod": "Clear widening - requires further imaging",
            "severe": "Marked widening - urgent evaluation for dissection or mass",
        },
        "urgency": {"low": [0.35, 0.65], "medium": [0.65, 0.8], "high": [0.8, 1.0]},
        "urgency_message": {
            "low": "Repeat PA chest X-ray and clinical review within 1 week",
            "medium": "Cardiology or medicine referral this week",
            "high": "Urgent referral today - exclude aortic dissection or mediastinal mass",
        },
        "specialist": "Cardiologist or General Physician",
        "differential": [
            "Cardiomegaly - enlarged heart alone",
            "Aortic aneurysm or dissection - requires urgent exclusion",
            "Mediastinal lymphadenopathy - TB, lymphoma, sarcoidosis",
            "Retrosternal goitre",
            "AP or rotated film - technical artefact, not pathology",
        ],
        "lab_tests": [
            "Repeat PA chest X-ray with correct positioning",
            "ECG",
            "Echocardiogram",
            "CT chest with contrast if dissection or mass suspected",
            "Full blood count and inflammatory markers",
        ],
        "treatments": [
            "Treatment depends entirely on the underlying cause",
            "No treatment should be started on this radiological finding alone",
            "Confirm the cause with echocardiography or CT before any therapy",
        ],
        "prescription_note": "No medication is indicated for this finding by itself. Prescribing must follow a confirmed underlying diagnosis.",
        "follow_up": "Repeat imaging and specialist review. Do not dismiss without establishing a cause.",
        "admission_criteria": "Admit immediately if tearing chest pain radiating to the back, unequal arm blood pressures, or signs of superior vena cava obstruction.",
        "prevention": [
            "Control blood pressure",
            "Ensure correct PA positioning when filming to avoid false positives",
        ],
        "emergency_signs": "Call emergency services for sudden tearing chest or back pain, unequal pulses between arms, fainting, or rapidly progressive facial and arm swelling.",
    },

    "Lung Opacity": {
        "full_name": "Lung Opacity",
        "icd10": "R91.8",
        "description": "A general increase in lung density. This is a descriptive finding with many possible causes - infection, fluid, collapse, scarring or mass - and it requires clinical correlation rather than treatment in its own right.",
        "imaging_findings": "Any area of lung appearing whiter than surrounding tissue. Distribution matters: patchy suggests infection, dependent and layered suggests fluid, wedge-shaped suggests collapse or infarction.",
        "common_symptoms": [
            "Cough",
            "Shortness of breath",
            "Fever if infective in origin",
            "Chest pain",
            "Sometimes entirely asymptomatic",
        ],
        "risk_factors": [
            "Recent respiratory infection",
            "Smoking",
            "Immunosuppression, including HIV and diabetes",
            "Heart failure",
            "Occupational dust exposure",
            "Tuberculosis contact",
        ],
        "severity": {
            "mild": "Small localised opacity",
            "mod": "Segmental or lobar involvement",
            "severe": "Extensive or bilateral opacity with hypoxia",
        },
        "urgency": {"low": [0.35, 0.65], "medium": [0.65, 0.8], "high": [0.8, 1.0]},
        "urgency_message": {
            "low": "Clinical correlation and review within 1 week",
            "medium": "Medical review within 48 hours",
            "high": "Same-day medical assessment - check oxygen saturation",
        },
        "specialist": "Pulmonologist or General Physician",
        "differential": [
            "Pneumonia - with fever and productive cough",
            "Pulmonary oedema - bilateral, often with cardiomegaly",
            "Atelectasis - with volume loss and mediastinal shift",
            "Tuberculosis - upper zone predominance, chronic symptoms",
            "Malignancy - persistent opacity that does not resolve",
        ],
        "lab_tests": [
            "Full blood count with differential",
            "Sputum for AFB and culture - essential where TB is endemic",
            "C-reactive protein",
            "Pulse oximetry",
            "Repeat chest X-ray after 4-6 weeks to confirm resolution",
        ],
        "treatments": [
            "Treat the underlying cause once identified",
            "Empirical antibiotics only where infection is clinically likely",
            "Supplemental oxygen if saturation is below 94 percent",
            "An opacity that fails to clear on repeat imaging needs CT and specialist review",
        ],
        "prescription_note": "Do not prescribe on this finding alone. Where pneumonia is clinically probable, follow local antibiotic guidance. In TB-endemic areas, exclude tuberculosis before assuming bacterial pneumonia.",
        "follow_up": "Repeat chest X-ray at 4-6 weeks is essential. A persistent opacity requires CT to exclude malignancy.",
        "admission_criteria": "Admit if oxygen saturation is below 92 percent, respiratory rate above 30, confusion, or inability to maintain oral intake.",
        "prevention": [
            "Stop smoking",
            "Pneumococcal and influenza vaccination",
            "Prompt treatment of respiratory infections",
        ],
        "emergency_signs": "Seek emergency care for severe breathlessness, blue lips or fingertips, confusion, or oxygen saturation below 90 percent.",
    },

    "Lung Lesion": {
        "full_name": "Lung Lesion",
        "icd10": "R91.1",
        "description": "A focal abnormality within the lung - a nodule, mass, or cavity. Because malignancy and tuberculosis both present this way, a lung lesion always requires follow-up and must never be assumed benign on a single film.",
        "imaging_findings": "A discrete rounded or irregular density. Spiculated margins, size above 3 cm, and upper lobe cavitation all raise concern. Compare with previous films - stability over two years is reassuring.",
        "common_symptoms": [
            "Often none - commonly found incidentally",
            "Persistent cough lasting more than 3 weeks",
            "Coughing blood",
            "Unintentional weight loss",
            "Chest pain",
            "Night sweats - consider tuberculosis",
        ],
        "risk_factors": [
            "Smoking - the dominant risk factor",
            "Age above 40",
            "Tuberculosis exposure",
            "Occupational asbestos or silica exposure",
            "Family history of lung cancer",
            "Biomass or cooking-smoke exposure",
        ],
        "severity": {
            "mild": "Under 8 mm, well-defined margins",
            "mod": "8-30 mm, requires CT characterisation",
            "severe": "Above 30 mm, or spiculated, or cavitating",
        },
        "urgency": {"low": [0.35, 0.65], "medium": [0.65, 0.8], "high": [0.8, 1.0]},
        "urgency_message": {
            "low": "CT chest and specialist review within 2-4 weeks",
            "medium": "CT chest and pulmonology referral within 1-2 weeks",
            "high": "Urgent referral - CT chest within days",
        },
        "specialist": "Pulmonologist - with oncology input if malignancy suspected",
        "differential": [
            "Bronchogenic carcinoma - must be excluded",
            "Tuberculoma - common in endemic regions, often upper lobe",
            "Benign granuloma - often calcified and stable",
            "Metastatic deposit - usually multiple and rounded",
            "Lung abscess - cavitating, with fever",
        ],
        "lab_tests": [
            "CT chest with contrast - the essential next step",
            "Sputum for AFB, three samples",
            "GeneXpert MTB/RIF where available",
            "Full blood count and ESR",
            "PET-CT or biopsy if malignancy suspected",
        ],
        "treatments": [
            "No treatment until the lesion is characterised",
            "Anti-tuberculous therapy only after microbiological confirmation",
            "Oncology referral if imaging or biopsy suggests malignancy",
            "Serial CT surveillance for small indeterminate nodules",
        ],
        "prescription_note": "Never start empirical treatment for a lung lesion. Both TB therapy and cancer treatment require confirmation - treating the wrong one causes real harm and delays the correct diagnosis.",
        "follow_up": "Mandatory. A lesion lost to follow-up is the commonest route to a late cancer diagnosis. Ensure the patient leaves with a dated appointment.",
        "admission_criteria": "Admit for significant haemoptysis, suspected superior vena cava obstruction, or severe breathlessness.",
        "prevention": [
            "Stop smoking - the single most effective measure",
            "Reduce indoor biomass smoke exposure",
            "Screen and treat household TB contacts",
        ],
        "emergency_signs": "Seek emergency care for coughing large volumes of blood, sudden severe breathlessness, or swelling of the face and neck.",
    },

    "Pleural Other": {
        "full_name": "Other Pleural Abnormality",
        "icd10": "J94.9",
        "description": "A pleural abnormality that is neither a simple effusion nor typical pleural thickening - for example plaques, calcification, or an irregular pleural-based density. Asbestos-related disease and pleural malignancy both present this way.",
        "imaging_findings": "Irregular pleural-based densities, calcified plaques (often along the diaphragm), or nodular pleural thickening. Nodularity and circumferential involvement raise concern for mesothelioma or metastatic disease.",
        "common_symptoms": [
            "Frequently asymptomatic",
            "Chest pain, often dull and persistent",
            "Breathlessness on exertion",
            "Reduced chest expansion on the affected side",
        ],
        "risk_factors": [
            "Asbestos exposure - construction, shipbuilding, brake linings",
            "Previous tuberculous pleurisy",
            "Previous empyema or haemothorax",
            "Prior chest radiotherapy",
            "Connective tissue disease",
        ],
        "severity": {
            "mild": "Isolated plaques, no functional impact",
            "mod": "Extensive plaques or thickening with mild restriction",
            "severe": "Nodular or circumferential thickening - exclude malignancy",
        },
        "urgency": {"low": [0.35, 0.65], "medium": [0.65, 0.8], "high": [0.8, 1.0]},
        "urgency_message": {
            "low": "Document occupational history; routine review",
            "medium": "CT chest and specialist referral within 2 weeks",
            "high": "Urgent CT and pulmonology referral - exclude mesothelioma",
        },
        "specialist": "Pulmonologist - with occupational medicine input where relevant",
        "differential": [
            "Asbestos-related pleural plaques - bilateral, calcified",
            "Mesothelioma - nodular, circumferential, usually unilateral",
            "Pleural metastases - often with a known primary",
            "Post-tuberculous or post-empyema fibrosis",
            "Loculated effusion",
        ],
        "lab_tests": [
            "CT chest - required to characterise pleural disease",
            "Detailed occupational and exposure history",
            "Spirometry to assess restriction",
            "Pleural biopsy if malignancy suspected",
        ],
        "treatments": [
            "Plaques alone need no treatment, only surveillance",
            "Treat the underlying cause where identified",
            "Manage pain symptomatically",
            "Refer for specialist assessment if nodular or progressive",
        ],
        "prescription_note": "No specific drug therapy. Analgesia for pleuritic pain as required.",
        "follow_up": "Long-term surveillance where there is asbestos exposure - mesothelioma can appear decades later.",
        "admission_criteria": "Admit for severe breathlessness or a large associated effusion requiring drainage.",
        "prevention": [
            "Occupational asbestos protection and monitoring",
            "Prompt and complete treatment of pleural infection",
        ],
        "emergency_signs": "Seek urgent care for rapidly worsening breathlessness or severe unremitting chest pain.",
    },

    "Fracture": {
        "full_name": "Fracture (Rib or Thoracic)",
        "icd10": "S22.9",
        "description": "A break in a rib, clavicle, or thoracic vertebra. Most isolated rib fractures heal without intervention; the important task is excluding associated injury to the lung, pleura, or abdominal organs.",
        "imaging_findings": "A cortical break or step in the rib contour. Check specifically for accompanying pneumothorax, haemothorax, or lung contusion. Three or more consecutive ribs fractured in two places indicates flail chest.",
        "common_symptoms": [
            "Sharp localised chest pain, worse on breathing",
            "Point tenderness over the fracture site",
            "Pain on coughing or movement",
            "Shallow breathing to limit pain",
            "Visible bruising",
        ],
        "risk_factors": [
            "Recent trauma, fall, or road accident",
            "Osteoporosis - fractures occur with minimal force",
            "Age above 65",
            "Chronic steroid use",
            "Bone metastases - pathological fracture",
            "Severe chronic cough",
        ],
        "severity": {
            "mild": "Single undisplaced rib fracture",
            "mod": "Multiple fractures, no complications",
            "severe": "Flail chest, or fracture with pneumothorax or haemothorax",
        },
        "urgency": {"low": [0.35, 0.65], "medium": [0.65, 0.8], "high": [0.8, 1.0]},
        "urgency_message": {
            "low": "Analgesia and breathing exercises; review in 1 week",
            "medium": "Same-week review - confirm no pneumothorax",
            "high": "Immediate assessment - exclude pneumothorax and internal injury",
        },
        "specialist": "Orthopaedics or Trauma - Cardiothoracic surgery if flail chest",
        "differential": [
            "Costochondritis - tender but no fracture line",
            "Pathological fracture - suspect if trauma was trivial",
            "Old healed fracture - callus, no acute tenderness",
            "Muscular strain",
        ],
        "lab_tests": [
            "Repeat chest X-ray to exclude delayed pneumothorax",
            "Pulse oximetry",
            "CT chest if multiple fractures or suspected internal injury",
            "Bone profile and DEXA if the fracture was low-impact",
        ],
        "treatments": [
            "Adequate analgesia - undertreated pain causes shallow breathing and pneumonia",
            "Deep breathing exercises and incentive spirometry",
            "Do NOT strap or bind the chest - it restricts ventilation",
            "Treat osteoporosis where the fracture was low-impact",
        ],
        "prescription_note": "Paracetamol and NSAIDs are first line. Pain control is the priority: patients who splint their breathing because of pain go on to develop atelectasis and pneumonia.",
        "follow_up": "Review at 1-2 weeks. Most rib fractures unite in 6 weeks. Investigate for underlying bone disease if the trauma was minor.",
        "admission_criteria": "Admit for flail chest, three or more rib fractures, associated pneumothorax or haemothorax, oxygen saturation below 92 percent, or inadequate pain control.",
        "prevention": [
            "Falls prevention in older adults",
            "Treat osteoporosis",
            "Vehicle seatbelt use",
        ],
        "emergency_signs": "Call emergency services for severe breathlessness, a chest wall moving paradoxically with breathing, coughing blood, or abdominal pain suggesting organ injury.",
    },

    "Support Devices": {
        "full_name": "Support Devices Present",
        "icd10": "Z95.9",
        "description": "Medical hardware is visible - for example an endotracheal tube, central venous line, chest drain, pacemaker, or surgical clips. THIS IS NOT A DISEASE. It is contextual information, and its clinical value lies in confirming correct device placement.",
        "imaging_findings": "Radio-opaque hardware. Verify position: endotracheal tube tip 2-6 cm above the carina; central line tip at the cavoatrial junction; nasogastric tube below the diaphragm and not coiled in the airway.",
        "common_symptoms": [],
        "risk_factors": [],
        "severity": {
            "mild": "Device present and correctly positioned",
            "mod": "Device present - position should be verified",
            "severe": "Device appears malpositioned - verify immediately",
        },
        # Urgency bands are set OUTSIDE the 0-1 probability range on purpose, so
        # no confidence level can ever classify this as medium or high urgency.
        # This is not a disease: a visible tube or pacemaker is context, never a
        # reason to send a patient to a district hospital. An earlier version
        # used normal bands and the app announced "Support Devices Present -
        # REFER URGENTLY" as a diagnosis.
        "urgency": {"low": [0.0, 1.01], "medium": [1.02, 1.03], "high": [1.04, 1.05]},
        "urgency_message": {
            "low": "Medical hardware visible - confirm correct placement against the clinical record. Not a diagnosis.",
            "medium": "Medical hardware visible - confirm correct placement.",
            "high": "Medical hardware visible - confirm correct placement.",
        },
        "specialist": "The team responsible for the device",
        "differential": [
            "External artefact - ECG leads, clothing, jewellery",
            "Retained surgical material",
        ],
        "lab_tests": ["Confirm device type and expected position from clinical notes"],
        "treatments": [
            "Not applicable - this finding requires no treatment",
            "Act only if the device appears malpositioned",
        ],
        "prescription_note": "Not applicable. This is not a diagnosis and no medication follows from it.",
        "follow_up": "As determined by the underlying reason the device was placed.",
        "admission_criteria": "Not applicable in itself. A malpositioned endotracheal tube or central line is an immediate clinical emergency and must be corrected at once.",
        "prevention": [],
        "emergency_signs": "If an airway device appears malpositioned in a breathless patient, seek immediate clinical assistance.",
    },

    "No Finding": {
        "full_name": "No Significant Abnormality Detected",
        "icd10": "Z13.89",
        "description": "No significant radiographic abnormality was detected. IMPORTANT: this is not the same as healthy. Chest X-rays miss early tuberculosis, small lesions hidden behind the heart or diaphragm, and many conditions that do not alter lung density at all. A normal film never overrides worrying symptoms.",
        "imaging_findings": "Clear lung fields, normal cardiothoracic ratio, sharp costophrenic angles, no visible focal lesion.",
        "common_symptoms": [],
        "risk_factors": [],
        "severity": {
            "mild": "No abnormality detected",
            "mod": "No abnormality detected",
            "severe": "No abnormality detected",
        },
        "urgency": {"low": [0.0, 1.01], "medium": [1.02, 1.03], "high": [1.04, 1.05]},
        "urgency_message": {
            "low": "No abnormality detected - correlate with symptoms",
            "medium": "No abnormality detected - correlate with symptoms",
            "high": "No abnormality detected - correlate with symptoms",
        },
        "specialist": "Not required on imaging grounds alone",
        "differential": [
            "Early disease not yet visible on plain film",
            "Lesion obscured behind heart, diaphragm, or clavicles",
            "Disease that does not alter radiographic density - asthma, early COPD, pulmonary embolism",
        ],
        "lab_tests": [
            "Guided by symptoms, not by this result",
            "In a symptomatic patient in a TB-endemic area, send sputum for AFB regardless of a normal film",
        ],
        "treatments": [
            "None indicated on imaging grounds",
            "Continue investigating clinically if symptoms persist",
        ],
        "prescription_note": "Not applicable.",
        "follow_up": "If symptoms persist or worsen, repeat imaging and investigate further. A normal chest X-ray does not exclude serious disease.",
        "admission_criteria": "Based on clinical condition, never on this imaging result.",
        "prevention": [],
        "emergency_signs": "A normal X-ray does not rule out an emergency. Severe breathlessness, chest pain, or coughing blood still require urgent assessment.",
    },
}


EXTENDED_CLINICAL_SOURCES = {
    "Enlarged Cardiomediastinum": {
        "citation": "Erbel R, et al. 2014 ESC Guidelines on the diagnosis and treatment of aortic diseases. Eur Heart J. 2014;35(41):2873-2926.",
        "doi": "10.1093/eurheartj/ehu281",
        "guideline_body": "European Society of Cardiology (ESC)",
        "key_criteria": "Mediastinal width >8 cm on PA film warrants further imaging. AP and supine projection magnify the mediastinum and are a common source of false positives.",
    },
    "Lung Opacity": {
        "citation": "Metlay JP, et al. Diagnosis and Treatment of Adults with Community-acquired Pneumonia. Am J Respir Crit Care Med. 2019;200(7):e45-e67.",
        "doi": "10.1164/rccm.201908-1581ST",
        "guideline_body": "American Thoracic Society / IDSA",
        "key_criteria": "Radiographic opacity plus compatible clinical features is required for a pneumonia diagnosis; imaging alone is insufficient. Repeat imaging at 4-6 weeks to confirm resolution.",
    },
    "Lung Lesion": {
        "citation": "MacMahon H, et al. Guidelines for Management of Incidental Pulmonary Nodules Detected on CT Images: From the Fleischner Society 2017. Radiology. 2017;284(1):228-243.",
        "doi": "10.1148/radiol.2017161659",
        "guideline_body": "Fleischner Society",
        "key_criteria": "Nodules >8 mm require CT characterisation and consideration of PET-CT or biopsy. In TB-endemic settings, exclude tuberculosis before assuming malignancy.",
    },
    "Pleural Other": {
        "citation": "Roberts ME, et al. British Thoracic Society Guideline for pleural disease. Thorax. 2023;78(Suppl 3):s1-s42.",
        "doi": "10.1136/thorax-2022-219784",
        "guideline_body": "British Thoracic Society (BTS)",
        "key_criteria": "Nodular or circumferential pleural thickening requires CT and specialist referral to exclude mesothelioma. Occupational exposure history is essential.",
    },
    "Fracture": {
        "citation": "Kessel B, et al. Rib fractures: comparison of associated injuries between pediatric and adult population. Am J Surg. 2014;208(5):831-834.",
        "doi": "10.1016/j.amjsurg.2013.10.033",
        "guideline_body": "General trauma literature",
        "key_criteria": "Adequate analgesia is the priority - inadequately treated rib fracture pain leads to splinting, atelectasis and pneumonia. Chest strapping is contraindicated.",
    },
    "Support Devices": {
        "citation": "Godoy MCB, et al. Chest radiography in the ICU: Part 1, Evaluation of airway, enteric, and pleural tubes. AJR Am J Roentgenol. 2012;198(3):563-571.",
        "doi": "10.2214/AJR.10.7226",
        "guideline_body": "American Roentgen Ray Society",
        "key_criteria": "Endotracheal tube tip should lie 2-6 cm above the carina. Central venous catheter tip at the cavoatrial junction. Verify placement on every film.",
    },
    "No Finding": {
        "citation": "World Health Organization. Chest radiography in tuberculosis detection: summary of current WHO recommendations. WHO, Geneva, 2016.",
        "doi": "",
        "guideline_body": "World Health Organization (WHO)",
        "key_criteria": "Chest radiography has limited sensitivity for early tuberculosis. A normal film does not exclude active TB in a symptomatic patient - sputum testing remains mandatory.",
    },
}


# ── fallback for genuinely unknown findings ─────────────────────────────────
# A new dataset can introduce a finding the knowledge base has never seen.
# Returning a safe, explicitly-unknown card beats a KeyError that takes down the
# whole report - and it tells the clinician the system has no guidance rather
# than inventing some.
UNKNOWN_TEMPLATE = {
    "full_name": None,                       # filled at call time
    "icd10": "R93.8",
    "description": "This finding is not yet in the AyShCXR clinical knowledge base. The AI has flagged it from the image, but no structured guidance is available.",
    "imaging_findings": "Refer to the Grad-CAM overlay for the region the model responded to.",
    "common_symptoms": [],
    "risk_factors": [],
    "severity": {"mild": "Not characterised", "mod": "Not characterised",
                 "severe": "Not characterised"},
    "urgency": {"low": [0.35, 0.65], "medium": [0.65, 0.8], "high": [0.8, 1.0]},
    "urgency_message": {
        "low": "Clinical correlation advised",
        "medium": "Clinical correlation advised - medical review this week",
        "high": "Clinical correlation advised - prompt medical review",
    },
    "specialist": "General Physician",
    "differential": [],
    "lab_tests": ["Guided by clinical assessment"],
    "treatments": ["No automated guidance available - assess clinically"],
    "prescription_note": "No medication guidance available for this finding. Do not prescribe on the AI output alone.",
    "follow_up": "Determined by clinical assessment.",
    "admission_criteria": "Based on the patient's clinical condition.",
    "prevention": [],
    "emergency_signs": "Standard emergency criteria apply - severe breathlessness, chest pain, coughing blood, or altered consciousness.",
}

# findings that are NOT pathologies: no urgency score, no treatment plan
NON_PATHOLOGY = ("Support Devices", "No Finding")


def install(disease_info: dict, clinical_sources: dict) -> int:
    """Merge the extension into medical_knowledge's dictionaries.

    Uses setdefault so an existing NIH-14 entry is NEVER overwritten.
    Returns the number of findings added.
    """
    added = 0
    for name, entry in EXTENDED_DISEASE_INFO.items():
        if name not in disease_info:
            disease_info[name] = entry
            added += 1
    for name, entry in EXTENDED_CLINICAL_SOURCES.items():
        clinical_sources.setdefault(name, entry)
    return added


def get_info_safe(disease_info: dict, disease: str) -> dict:
    """DISEASE_INFO lookup that never raises."""
    if disease in disease_info:
        return disease_info[disease]
    entry = dict(UNKNOWN_TEMPLATE)
    entry["full_name"] = f"{disease} (not in knowledge base)"
    return entry


def is_pathology(disease: str) -> bool:
    return disease not in NON_PATHOLOGY


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import medical_knowledge as mk

    n = install(mk.DISEASE_INFO, mk.DISEASE_CLINICAL_SOURCES)
    print("=" * 68)
    print("  medical_knowledge extension self-test")
    print("=" * 68)
    print(f"\nadded {n} findings -> DISEASE_INFO now has {len(mk.DISEASE_INFO)}")

    # schema must match the existing NIH-14 entries exactly
    ref = set(mk.DISEASE_INFO["Cardiomegaly"])
    bad = []
    for name in EXTENDED_DISEASE_INFO:
        got = set(mk.DISEASE_INFO[name])
        if got != ref:
            bad.append((name, ref - got, got - ref))
    print(f"schema match vs Cardiomegaly ({len(ref)} keys): "
          f"{'OK' if not bad else bad}")
    assert not bad, "schema mismatch"

    # every canonical finding must resolve
    try:
        from disease_ontology import CANONICAL_ORDER
        missing = [d for d in CANONICAL_ORDER if d not in mk.DISEASE_INFO]
        print(f"canonical coverage: {len(CANONICAL_ORDER) - len(missing)}"
              f"/{len(CANONICAL_ORDER)}  missing={missing or 'none'}")
        assert not missing
    except ImportError:
        print("disease_ontology not importable - skipped coverage check")

    print(f"\nnon-pathology flags:")
    for d in ("Support Devices", "No Finding", "Pneumonia"):
        print(f"   {d:18s} is_pathology={is_pathology(d)}")

    print("\nfallback for an unseen finding:")
    u = get_info_safe(mk.DISEASE_INFO, "Aortic Enlargement")
    print(f"   full_name : {u['full_name']}")
    print(f"   treatments: {u['treatments'][0]}")
    assert "not in knowledge base" in u["full_name"]

    print("\n" + "=" * 68)
    print("  ALL CHECKS PASSED")
    print("=" * 68)
