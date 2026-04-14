# stage2_questions.py
# AyShCXR — Two-Stage Clinical Decision System
# Dynamic question bank for Stage 2 targeted narrowing
# Each disease has distinguishing questions with scoring weights

STAGE2_QUESTIONS = {

    "Pneumonia": {
        "questions": [
            {
                "id": "pneu_fever",
                "text": "Is there fever above 38.5°C with chills or rigors?",
                "yes_score": 0.9,
                "no_score": -0.3,
                "distinguishes_from": ["Effusion", "Fibrosis", "Atelectasis"]
            },
            {
                "id": "pneu_sputum",
                "text": "Is there productive cough with yellow or green sputum?",
                "yes_score": 0.8,
                "no_score": -0.2,
                "distinguishes_from": ["Nodule", "Mass", "Fibrosis"]
            },
            {
                "id": "pneu_sudden",
                "text": "Did symptoms start suddenly within hours to days?",
                "yes_score": 0.7,
                "no_score": -0.1,
                "distinguishes_from": ["Fibrosis", "Emphysema", "Cardiomegaly"]
            },
            {
                "id": "pneu_pleuritic",
                "text": "Is there sharp chest pain that worsens on breathing or coughing?",
                "yes_score": 0.6,
                "no_score": 0.0,
                "distinguishes_from": ["Edema", "Cardiomegaly"]
            },
            {
                "id": "pneu_response",
                "text": "Has any antibiotic course been started in the last 2 weeks?",
                "yes_score": 0.5,
                "no_score": 0.0,
                "distinguishes_from": []
            }
        ]
    },

    "Effusion": {
        "questions": [
            {
                "id": "effu_lieflat",
                "text": "Is it difficult or impossible to lie flat due to breathlessness?",
                "yes_score": 0.9,
                "no_score": -0.2,
                "distinguishes_from": ["Pneumonia", "Pneumothorax", "Atelectasis"]
            },
            {
                "id": "effu_swelling",
                "text": "Is there swelling of legs or ankles that pits on pressing?",
                "yes_score": 0.8,
                "no_score": -0.1,
                "distinguishes_from": ["Pneumonia", "Mass", "Nodule"]
            },
            {
                "id": "effu_gradual",
                "text": "Did breathlessness develop gradually over days to weeks?",
                "yes_score": 0.7,
                "no_score": -0.2,
                "distinguishes_from": ["Pneumothorax", "Pneumonia"]
            },
            {
                "id": "effu_dull",
                "text": "Is the breathlessness worse on one specific side of the chest?",
                "yes_score": 0.6,
                "no_score": 0.0,
                "distinguishes_from": ["Edema"]
            },
            {
                "id": "effu_cardiac",
                "text": "Is there known heart failure, liver disease, or kidney disease?",
                "yes_score": 0.7,
                "no_score": 0.0,
                "distinguishes_from": ["Pneumonia", "Mass"]
            }
        ]
    },

    "Atelectasis": {
        "questions": [
            {
                "id": "atel_surgery",
                "text": "Was there any surgery or general anaesthesia in the last 4 weeks?",
                "yes_score": 0.9,
                "no_score": -0.1,
                "distinguishes_from": ["Pneumonia", "Effusion", "Mass"]
            },
            {
                "id": "atel_bedrest",
                "text": "Has the patient been on prolonged bed rest or immobile?",
                "yes_score": 0.7,
                "no_score": -0.1,
                "distinguishes_from": ["Pneumothorax", "Mass"]
            },
            {
                "id": "atel_shallow",
                "text": "Is breathing rapid and shallow rather than deep?",
                "yes_score": 0.6,
                "no_score": 0.0,
                "distinguishes_from": ["Effusion", "Fibrosis"]
            },
            {
                "id": "atel_nofever",
                "text": "Is there absence of fever or minimal fever only?",
                "yes_score": 0.5,
                "no_score": -0.3,
                "distinguishes_from": ["Pneumonia", "Consolidation"]
            },
            {
                "id": "atel_mucus",
                "text": "Is there known COPD, asthma, or history of mucus plugging?",
                "yes_score": 0.6,
                "no_score": 0.0,
                "distinguishes_from": []
            }
        ]
    },

    "Consolidation": {
        "questions": [
            {
                "id": "cons_fever",
                "text": "Is there high fever above 38.5°C with rigors?",
                "yes_score": 0.8,
                "no_score": -0.2,
                "distinguishes_from": ["Fibrosis", "Atelectasis", "Effusion"]
            },
            {
                "id": "cons_lobar",
                "text": "Is the breathlessness confined to one lobe area of chest?",
                "yes_score": 0.7,
                "no_score": 0.0,
                "distinguishes_from": ["Edema", "Infiltration"]
            },
            {
                "id": "cons_percussion",
                "text": "Does tapping the chest produce a dull sound on one side?",
                "yes_score": 0.7,
                "no_score": -0.1,
                "distinguishes_from": ["Pneumothorax", "Emphysema"]
            },
            {
                "id": "cons_rusty",
                "text": "Is the sputum rust-coloured or blood-tinged?",
                "yes_score": 0.8,
                "no_score": 0.0,
                "distinguishes_from": ["Effusion", "Atelectasis"]
            },
            {
                "id": "cons_diabetes",
                "text": "Is there diabetes mellitus or other immunosuppression?",
                "yes_score": 0.6,
                "no_score": 0.0,
                "distinguishes_from": []
            }
        ]
    },

    "Cardiomegaly": {
        "questions": [
            {
                "id": "card_exertion",
                "text": "Does breathlessness occur mainly on exertion or climbing stairs?",
                "yes_score": 0.8,
                "no_score": -0.1,
                "distinguishes_from": ["Pneumonia", "Pneumothorax"]
            },
            {
                "id": "card_pnd",
                "text": "Does the patient wake from sleep breathless and need to sit up?",
                "yes_score": 0.9,
                "no_score": -0.1,
                "distinguishes_from": ["Pneumonia", "Mass", "Fibrosis"]
            },
            {
                "id": "card_htn",
                "text": "Is there known hypertension or high blood pressure history?",
                "yes_score": 0.7,
                "no_score": 0.0,
                "distinguishes_from": ["Pneumonia", "Fibrosis"]
            },
            {
                "id": "card_palpitations",
                "text": "Are there palpitations or awareness of irregular heartbeat?",
                "yes_score": 0.7,
                "no_score": 0.0,
                "distinguishes_from": ["Effusion", "Pneumonia"]
            },
            {
                "id": "card_gradual",
                "text": "Has breathlessness been worsening gradually over months?",
                "yes_score": 0.6,
                "no_score": -0.2,
                "distinguishes_from": ["Pneumothorax", "Pneumonia"]
            }
        ]
    },

    "Edema": {
        "questions": [
            {
                "id": "edem_bilateral",
                "text": "Is breathlessness affecting both sides equally and is worse lying flat?",
                "yes_score": 0.9,
                "no_score": -0.2,
                "distinguishes_from": ["Effusion", "Pneumonia", "Atelectasis"]
            },
            {
                "id": "edem_frothy",
                "text": "Is there any pink or frothy sputum when coughing?",
                "yes_score": 1.0,
                "no_score": 0.0,
                "distinguishes_from": ["Pneumonia", "Fibrosis", "Effusion"]
            },
            {
                "id": "edem_cardiac",
                "text": "Is there known heart failure or recent heart attack?",
                "yes_score": 0.9,
                "no_score": -0.1,
                "distinguishes_from": ["Pneumonia", "Fibrosis"]
            },
            {
                "id": "edem_rapid",
                "text": "Did breathlessness develop very rapidly over hours?",
                "yes_score": 0.8,
                "no_score": -0.1,
                "distinguishes_from": ["Fibrosis", "Effusion"]
            },
            {
                "id": "edem_wheeze",
                "text": "Is there wheeze or bubbling sound when breathing?",
                "yes_score": 0.6,
                "no_score": 0.0,
                "distinguishes_from": ["Pneumothorax", "Atelectasis"]
            }
        ]
    },

    "Emphysema": {
        "questions": [
            {
                "id": "emph_smoking",
                "text": "Is there a smoking history of more than 10 pack-years?",
                "yes_score": 0.9,
                "no_score": -0.3,
                "distinguishes_from": ["Effusion", "Pneumonia", "Cardiomegaly"]
            },
            {
                "id": "emph_chronic",
                "text": "Has breathlessness been present and worsening for more than 2 years?",
                "yes_score": 0.8,
                "no_score": -0.3,
                "distinguishes_from": ["Pneumonia", "Pneumothorax", "Atelectasis"]
            },
            {
                "id": "emph_barrel",
                "text": "Does the chest appear barrel-shaped or over-expanded?",
                "yes_score": 0.8,
                "no_score": -0.1,
                "distinguishes_from": ["Effusion", "Cardiomegaly"]
            },
            {
                "id": "emph_pursed",
                "text": "Does the patient breathe out slowly through pursed lips?",
                "yes_score": 0.7,
                "no_score": 0.0,
                "distinguishes_from": ["Pneumonia", "Effusion"]
            },
            {
                "id": "emph_occupation",
                "text": "Is there history of coal mining, textile, or dust-heavy occupation?",
                "yes_score": 0.6,
                "no_score": 0.0,
                "distinguishes_from": []
            }
        ]
    },

    "Fibrosis": {
        "questions": [
            {
                "id": "fibr_velcro",
                "text": "Are there crackling sounds at lung bases like Velcro being pulled apart?",
                "yes_score": 0.9,
                "no_score": -0.2,
                "distinguishes_from": ["Emphysema", "Effusion", "Pneumothorax"]
            },
            {
                "id": "fibr_clubbing",
                "text": "Are the fingertips club-shaped or broader than normal?",
                "yes_score": 0.8,
                "no_score": -0.1,
                "distinguishes_from": ["Cardiomegaly", "Emphysema", "Effusion"]
            },
            {
                "id": "fibr_progressive",
                "text": "Has breathlessness been slowly progressive over years not months?",
                "yes_score": 0.8,
                "no_score": -0.2,
                "distinguishes_from": ["Pneumonia", "Atelectasis", "Pneumothorax"]
            },
            {
                "id": "fibr_drycough",
                "text": "Is there a persistent dry non-productive cough present for months?",
                "yes_score": 0.7,
                "no_score": -0.1,
                "distinguishes_from": ["Pneumonia", "Consolidation"]
            },
            {
                "id": "fibr_occupation",
                "text": "Is there history of asbestos, silica, or heavy dust exposure?",
                "yes_score": 0.7,
                "no_score": 0.0,
                "distinguishes_from": []
            }
        ]
    },

    "Mass": {
        "questions": [
            {
                "id": "mass_haemopt",
                "text": "Is there blood in the sputum or coughing up blood?",
                "yes_score": 1.0,
                "no_score": -0.1,
                "distinguishes_from": ["Effusion", "Cardiomegaly", "Atelectasis"]
            },
            {
                "id": "mass_weightloss",
                "text": "Is there unexplained weight loss of more than 5kg in 3 months?",
                "yes_score": 0.9,
                "no_score": -0.2,
                "distinguishes_from": ["Effusion", "Cardiomegaly", "Emphysema"]
            },
            {
                "id": "mass_smoking",
                "text": "Is there significant smoking history of more than 20 pack-years?",
                "yes_score": 0.8,
                "no_score": -0.1,
                "distinguishes_from": ["Effusion", "Fibrosis"]
            },
            {
                "id": "mass_hoarse",
                "text": "Has the voice become hoarse or changed in quality recently?",
                "yes_score": 0.8,
                "no_score": 0.0,
                "distinguishes_from": ["Effusion", "Pneumonia", "Atelectasis"]
            },
            {
                "id": "mass_age",
                "text": "Is the patient above 50 years of age?",
                "yes_score": 0.6,
                "no_score": -0.1,
                "distinguishes_from": []
            }
        ]
    },

    "Nodule": {
        "questions": [
            {
                "id": "nodu_asymp",
                "text": "Was this finding incidental with no respiratory symptoms at all?",
                "yes_score": 0.7,
                "no_score": -0.1,
                "distinguishes_from": ["Pneumonia", "Consolidation"]
            },
            {
                "id": "nodu_smoking",
                "text": "Is there any smoking history current or past?",
                "yes_score": 0.7,
                "no_score": -0.2,
                "distinguishes_from": []
            },
            {
                "id": "nodu_prevtb",
                "text": "Is there history of previous tuberculosis or fungal infection?",
                "yes_score": 0.6,
                "no_score": 0.0,
                "distinguishes_from": ["Mass"]
            },
            {
                "id": "nodu_age",
                "text": "Is the patient above 35 years of age?",
                "yes_score": 0.5,
                "no_score": -0.1,
                "distinguishes_from": []
            },
            {
                "id": "nodu_family",
                "text": "Is there family history of lung cancer?",
                "yes_score": 0.6,
                "no_score": 0.0,
                "distinguishes_from": []
            }
        ]
    },

    "Pneumothorax": {
        "questions": [
            {
                "id": "ptx_sudden",
                "text": "Did chest pain and breathlessness start suddenly and without warning?",
                "yes_score": 0.9,
                "no_score": -0.4,
                "distinguishes_from": ["Effusion", "Fibrosis", "Emphysema"]
            },
            {
                "id": "ptx_onesided",
                "text": "Is the pain and discomfort clearly on one side only?",
                "yes_score": 0.8,
                "no_score": -0.3,
                "distinguishes_from": ["Edema", "Effusion"]
            },
            {
                "id": "ptx_young",
                "text": "Is the patient a tall thin young male under 35 years?",
                "yes_score": 0.7,
                "no_score": 0.0,
                "distinguishes_from": ["Cardiomegaly", "Fibrosis"]
            },
            {
                "id": "ptx_copd",
                "text": "Is there known COPD, emphysema, or bullous lung disease?",
                "yes_score": 0.7,
                "no_score": 0.0,
                "distinguishes_from": []
            },
            {
                "id": "ptx_absent",
                "text": "Are breath sounds reduced or absent on one side of chest?",
                "yes_score": 0.8,
                "no_score": -0.2,
                "distinguishes_from": ["Cardiomegaly", "Emphysema"]
            }
        ]
    },

    "Infiltration": {
        "questions": [
            {
                "id": "infi_tbcontact",
                "text": "Is there known contact with a tuberculosis patient?",
                "yes_score": 0.9,
                "no_score": 0.0,
                "distinguishes_from": ["Effusion", "Cardiomegaly"]
            },
            {
                "id": "infi_nightsweat",
                "text": "Are there drenching night sweats soaking the clothes?",
                "yes_score": 0.8,
                "no_score": -0.1,
                "distinguishes_from": ["Effusion", "Atelectasis", "Emphysema"]
            },
            {
                "id": "infi_weightloss",
                "text": "Is there weight loss combined with persistent cough over weeks?",
                "yes_score": 0.8,
                "no_score": -0.1,
                "distinguishes_from": ["Atelectasis", "Effusion"]
            },
            {
                "id": "infi_chronic",
                "text": "Has the cough been present for more than 3 weeks without improvement?",
                "yes_score": 0.7,
                "no_score": -0.2,
                "distinguishes_from": ["Pneumonia", "Pneumothorax"]
            },
            {
                "id": "infi_hiv",
                "text": "Is there known HIV infection or significant immunosuppression?",
                "yes_score": 0.7,
                "no_score": 0.0,
                "distinguishes_from": []
            }
        ]
    },

    "Hernia": {
        "questions": [
            {
                "id": "hern_postmeal",
                "text": "Is breathlessness or chest discomfort worse specifically after meals?",
                "yes_score": 0.9,
                "no_score": -0.3,
                "distinguishes_from": ["Pneumonia", "Effusion", "Pneumothorax"]
            },
            {
                "id": "hern_heartburn",
                "text": "Is there heartburn, acid reflux, or regurgitation of food?",
                "yes_score": 0.8,
                "no_score": -0.2,
                "distinguishes_from": ["Cardiomegaly", "Fibrosis", "Emphysema"]
            },
            {
                "id": "hern_dysphagia",
                "text": "Is there difficulty swallowing solid food or a sensation of food sticking?",
                "yes_score": 0.7,
                "no_score": 0.0,
                "distinguishes_from": ["Effusion", "Pneumonia"]
            },
            {
                "id": "hern_obese",
                "text": "Is the patient obese or significantly overweight?",
                "yes_score": 0.6,
                "no_score": 0.0,
                "distinguishes_from": []
            },
            {
                "id": "hern_upright",
                "text": "Does sitting upright or standing relieve the chest discomfort?",
                "yes_score": 0.7,
                "no_score": -0.1,
                "distinguishes_from": ["Cardiomegaly", "Effusion"]
            }
        ]
    },

    "Pleural Thickening": {
        "questions": [
            {
                "id": "plth_asbestos",
                "text": "Is there occupational history of asbestos, shipyard, or demolition work?",
                "yes_score": 0.9,
                "no_score": -0.2,
                "distinguishes_from": ["Pneumonia", "Effusion", "Cardiomegaly"]
            },
            {
                "id": "plth_previnfect",
                "text": "Is there history of previous pleural infection or tuberculosis?",
                "yes_score": 0.8,
                "no_score": -0.1,
                "distinguishes_from": ["Emphysema", "Fibrosis"]
            },
            {
                "id": "plth_dull",
                "text": "Is breathlessness mild and on exertion only with no acute symptoms?",
                "yes_score": 0.6,
                "no_score": -0.2,
                "distinguishes_from": ["Pneumonia", "Pneumothorax", "Edema"]
            },
            {
                "id": "plth_chest_wall",
                "text": "Is there dull ache along the chest wall that is chronic and persistent?",
                "yes_score": 0.6,
                "no_score": 0.0,
                "distinguishes_from": []
            },
            {
                "id": "plth_surgery",
                "text": "Is there history of previous thoracic surgery or chest trauma?",
                "yes_score": 0.6,
                "no_score": 0.0,
                "distinguishes_from": []
            }
        ]
    }
}


def get_stage2_questions_for_diseases(top_diseases):
    """
    Given a list of top disease names from Stage 1,
    returns a deduplicated list of the most discriminating
    questions for those specific diseases.
    Each question tagged with which disease it helps identify.
    Max 3 questions per disease = max ~9-12 questions total.
    """
    selected = []
    seen_ids = set()

    for disease in top_diseases:
        if disease not in STAGE2_QUESTIONS:
            continue
        questions = STAGE2_QUESTIONS[disease]["questions"]
        # Pick top 3 questions for this disease
        for q in questions[:3]:
            if q["id"] not in seen_ids:
                seen_ids.add(q["id"])
                selected.append({
                    "id":           q["id"],
                    "text":         q["text"],
                    "for_disease":  disease,
                    "yes_score":    q["yes_score"],
                    "no_score":     q["no_score"],
                })

    return selected


def apply_stage2_scores(predictions, stage2_answers, top_diseases):
    """
    Applies Stage 2 targeted question answers to re-score
    the top diseases and pick one primary finding.

    predictions   : list of dicts with disease + probability
    stage2_answers: dict {question_id: True/False}
    top_diseases  : list of disease names that reached Stage 2

    Returns predictions with stage2_score added,
    and primary_disease name.

    Weighting:
      80% image AI score
      10% Stage 1 boost (already in probability)
      10% Stage 2 targeted questions
    """
    # Build disease -> question mapping
    disease_question_map = {}
    for disease in top_diseases:
        if disease not in STAGE2_QUESTIONS:
            continue
        disease_question_map[disease] = STAGE2_QUESTIONS[disease]["questions"]

    updated = []
    for pred in predictions:
        disease = pred["disease"]
        base_prob = pred["probability"]  # already has stage1 boost (80%+10%)

        if disease not in top_diseases:
            # Not a top disease — pass through unchanged
            pred["stage2_score"] = base_prob
            pred["stage2_delta"] = 0.0
            pred["is_primary"]   = False
            updated.append(pred)
            continue

        # Calculate Stage 2 evidence score for this disease
        questions = disease_question_map.get(disease, [])
        evidence_score = 0.0
        answered = 0

        for q in questions:
            q_id = q["id"]
            if q_id in stage2_answers:
                answered += 1
                if stage2_answers[q_id]:
                    evidence_score += q["yes_score"]
                else:
                    evidence_score += q["no_score"]

        # Normalise evidence score to 0-1 range
        if answered > 0:
            max_possible = sum(q["yes_score"] for q in questions[:answered])
            if max_possible > 0:
                evidence_score = max(0, evidence_score / max_possible)
            else:
                evidence_score = 0.5
        else:
            evidence_score = 0.5  # neutral if no questions answered

        # Final score: 90% existing (image+stage1) + 10% stage2
        stage2_contribution = evidence_score * 0.10
        final_score = (base_prob * 0.90) + stage2_contribution

        new_pred = dict(pred)
        new_pred["stage2_score"] = round(min(final_score, 0.99), 4)
        new_pred["stage2_delta"] = round(stage2_contribution, 4)
        new_pred["is_primary"]   = False
        updated.append(new_pred)

    # Mark primary disease — highest stage2_score among top diseases
    top_candidates = [p for p in updated if p["disease"] in top_diseases]
    if top_candidates:
        primary = max(top_candidates, key=lambda x: x["stage2_score"])
        primary["is_primary"] = True

    return updated