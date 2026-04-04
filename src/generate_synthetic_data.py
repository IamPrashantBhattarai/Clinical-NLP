"""
generate_synthetic_data.py
Purpose: Generate synthetic clinical data that mirrors the MIMIC-IV schema.
Use this for development and testing when real MIMIC-IV data is not available.

Outputs:
- data/synthetic_discharge.csv   : 1000 synthetic discharge summaries
- data/synthetic_admissions.csv  : Matching admissions table
- data/synthetic_patients.csv    : Matching patients table
"""

import random
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Clinical text building blocks
# ---------------------------------------------------------------------------

CHIEF_COMPLAINTS = [
    "shortness of breath", "chest pain", "altered mental status",
    "hypotension", "acute kidney injury", "sepsis", "COPD exacerbation",
    "heart failure exacerbation", "diabetic ketoacidosis", "stroke",
    "pneumonia", "GI bleed", "acute MI", "pulmonary embolism",
    "seizure", "abdominal pain", "fever and chills", "syncope",
    "uncontrolled hypertension", "acute liver failure",
]

PAST_MEDICAL_CONDITIONS = [
    "hypertension", "diabetes mellitus type 2", "chronic kidney disease",
    "coronary artery disease", "heart failure with reduced ejection fraction",
    "atrial fibrillation", "COPD", "obstructive sleep apnea",
    "hyperlipidemia", "peripheral vascular disease", "hypothyroidism",
    "chronic liver disease", "history of stroke", "depression",
    "anxiety disorder", "GERD", "osteoarthritis", "anemia",
    "history of DVT", "history of pulmonary embolism",
]

SOCIAL_HISTORY_TEMPLATES = [
    "The patient is a {age}-year-old {gender} who lives {living_situation}. "
    "{tobacco} {alcohol} {drugs} Works as {occupation}.",
]

LIVING_SITUATIONS = [
    "at home with family", "alone", "in an assisted living facility",
    "with a spouse", "independently",
]

OCCUPATIONS = [
    "a retired teacher", "a construction worker", "an accountant",
    "a homemaker", "a truck driver", "a nurse", "a farmer",
    "a retired military officer", "a store clerk", "unemployed",
]

TOBACCO = [
    "Former smoker, quit 10 years ago.", "Current smoker, 1 ppd.",
    "Never smoker.", "Former smoker, 30 pack-year history.",
]

ALCOHOL = [
    "Occasional alcohol use.", "Denies alcohol use.",
    "Heavy alcohol use, approximately 6 drinks per day.",
    "Social drinker.",
]

DRUGS = [
    "Denies illicit drug use.", "Former IV drug user.",
    "Occasional marijuana use.",
]

PHYSICAL_EXAM_VITALS = [
    "T {temp}°F, HR {hr} bpm, BP {sbp}/{dbp} mmHg, RR {rr} breaths/min, O2 sat {o2}% on {o2_device}.",
]

EXAM_FINDINGS = [
    "General: Alert and oriented, in mild distress.",
    "General: Lethargic, difficult to arouse.",
    "General: Alert and oriented x3, in no acute distress.",
    "Cardiovascular: Regular rate and rhythm, no murmurs.",
    "Cardiovascular: Irregularly irregular rhythm, 2+ pitting edema bilateral lower extremities.",
    "Pulmonary: Clear to auscultation bilaterally.",
    "Pulmonary: Diffuse wheezing and prolonged expiratory phase.",
    "Pulmonary: Decreased breath sounds at right base with dullness to percussion.",
    "Abdomen: Soft, non-tender, non-distended. Bowel sounds present.",
    "Abdomen: Tender in right upper quadrant. Positive Murphy sign.",
    "Extremities: No cyanosis or clubbing. Trace edema bilateral ankles.",
    "Neurological: AAOx3. Cranial nerves intact. No focal deficits.",
    "Neurological: Left-sided weakness. Speech slurred.",
]

HOSPITAL_COURSE_TEMPLATES = {
    "heart failure": (
        "The patient was admitted for acute decompensated heart failure. "
        "IV diuresis was initiated with furosemide. "
        "Echocardiogram showed EF of {ef}%. "
        "The patient was diuresed of approximately {diuresis}L over the hospitalization. "
        "BNP trended from {bnp_admit} to {bnp_discharge}. "
        "The patient was transitioned to oral diuretics and discharged with close follow-up "
        "with outpatient cardiology. {complication}"
    ),
    "pneumonia": (
        "The patient was admitted for community-acquired pneumonia. "
        "Chest X-ray confirmed {lobe} lobe infiltrate. "
        "Blood cultures were {culture_result}. Sputum culture grew {sputum}. "
        "IV antibiotics were started with {abx}. "
        "The patient defervesced by hospital day {defervescence_day}. "
        "Oxygen requirements improved and the patient was weaned to room air. "
        "{complication}"
    ),
    "sepsis": (
        "The patient was admitted to the ICU for sepsis secondary to {source}. "
        "Sepsis protocol was initiated. Blood cultures drawn and IV fluids administered. "
        "Vasopressors were required for hemodynamic support. "
        "Source was identified as {source}. Antibiotics were de-escalated per culture data. "
        "The patient was weaned off vasopressors by hospital day {vasopressor_days}. "
        "{complication}"
    ),
    "copd": (
        "The patient presented with acute exacerbation of COPD. "
        "Supplemental oxygen was initiated. "
        "Bronchodilators including albuterol and ipratropium nebulizations were administered. "
        "IV solumedrol was started and transitioned to oral prednisone. "
        "Antibiotics were added given increased sputum production. "
        "The patient was monitored on telemetry throughout the hospitalization. "
        "{complication}"
    ),
    "aki": (
        "The patient was admitted for acute kidney injury, likely {etiology}. "
        "IV fluids were administered for volume resuscitation. "
        "Nephrotoxic medications were held. "
        "Creatinine trended from {cr_admit} to {cr_discharge}. "
        "Renal ultrasound was {ultrasound_result}. "
        "Nephrology was consulted. {complication}"
    ),
    "stroke": (
        "The patient presented with acute ischemic stroke. "
        "CT head was negative for hemorrhage. "
        "MRI brain confirmed {location} infarct. "
        "tPA was {tpa_decision}. "
        "Antiplatelet therapy was initiated. "
        "Neurology was consulted and stroke workup initiated. "
        "Physical therapy and occupational therapy were involved. "
        "{complication}"
    ),
    "gi_bleed": (
        "The patient presented with {bleed_type} GI bleed. "
        "Hemoglobin on admission was {hgb_admit}. "
        "Two large-bore IV access placed and IV fluids administered. "
        "GI was consulted and {procedure} was performed showing {finding}. "
        "The patient was transfused {units} units of packed red blood cells. "
        "PPI infusion was started. {complication}"
    ),
    "dka": (
        "The patient was admitted for diabetic ketoacidosis with pH {ph} and glucose {glucose}. "
        "Insulin drip was started per DKA protocol. "
        "Aggressive IV fluid resuscitation was performed. "
        "Potassium was repleted as needed. "
        "The patient was transitioned to subcutaneous insulin when anion gap closed. "
        "Endocrinology was consulted for insulin regimen optimization. "
        "{complication}"
    ),
}

COMPLICATIONS = [
    "", "", "", "",  # Most have no complication
    "Course was complicated by atrial fibrillation with rapid ventricular response, rate controlled with IV metoprolol.",
    "Course was complicated by acute kidney injury, managed with IV fluids.",
    "Course was complicated by hospital-acquired pneumonia, treated with antibiotics.",
    "Course was complicated by delirium, managed with reorientation and low-dose haloperidol.",
    "Course was complicated by deep vein thrombosis, anticoagulation was initiated.",
]

DISCHARGE_MEDICATIONS = [
    "Lisinopril 10mg daily", "Metoprolol succinate 50mg daily",
    "Furosemide 40mg daily", "Atorvastatin 40mg nightly",
    "Aspirin 81mg daily", "Clopidogrel 75mg daily",
    "Metformin 1000mg twice daily", "Insulin glargine 20 units nightly",
    "Warfarin 5mg daily", "Apixaban 5mg twice daily",
    "Amlodipine 5mg daily", "Omeprazole 20mg daily",
    "Albuterol inhaler PRN", "Tiotropium inhaler daily",
    "Prednisone 40mg daily x5 days", "Azithromycin 250mg daily x5 days",
    "Levofloxacin 750mg daily x7 days", "Trimethoprim-sulfamethoxazole DS twice daily",
]

DISCHARGE_DISPOSITIONS = [
    "Home with home health services",
    "Home without services",
    "Skilled nursing facility",
    "Rehabilitation facility",
    "Long-term acute care hospital",
    "Hospice care",
]


# ---------------------------------------------------------------------------
# Helper generators
# ---------------------------------------------------------------------------

def _random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def _generate_vitals() -> str:
    temp = round(random.uniform(97.0, 103.5), 1)
    hr = random.randint(55, 140)
    sbp = random.randint(80, 200)
    dbp = random.randint(50, 110)
    rr = random.randint(12, 35)
    o2 = random.randint(82, 100)
    o2_device = random.choice(["room air", "2L NC", "4L NC", "6L NC", "high-flow nasal cannula", "non-rebreather mask"])
    return PHYSICAL_EXAM_VITALS[0].format(
        temp=temp, hr=hr, sbp=sbp, dbp=dbp, rr=rr, o2=o2, o2_device=o2_device
    )


def _generate_hospital_course(condition: str) -> str:
    template_key = condition.lower().replace(" ", "_")
    if template_key not in HOSPITAL_COURSE_TEMPLATES:
        template_key = random.choice(list(HOSPITAL_COURSE_TEMPLATES.keys()))

    template = HOSPITAL_COURSE_TEMPLATES[template_key]
    complication = random.choice(COMPLICATIONS)

    fills = {
        "ef": random.randint(15, 55),
        "diuresis": round(random.uniform(2, 8), 1),
        "bnp_admit": random.randint(500, 5000),
        "bnp_discharge": random.randint(100, 800),
        "lobe": random.choice(["right lower", "left lower", "right upper", "bilateral"]),
        "culture_result": random.choice(["negative", "positive for Streptococcus pneumoniae", "pending"]),
        "sputum": random.choice(["normal respiratory flora", "Pseudomonas aeruginosa", "Klebsiella pneumoniae"]),
        "abx": random.choice(["ceftriaxone and azithromycin", "piperacillin-tazobactam", "vancomycin and cefepime"]),
        "defervescence_day": random.randint(2, 5),
        "source": random.choice(["urinary tract infection", "pneumonia", "intra-abdominal source", "skin and soft tissue"]),
        "vasopressor_days": random.randint(2, 7),
        "etiology": random.choice(["prerenal azotemia", "contrast-induced nephropathy", "ATN"]),
        "cr_admit": round(random.uniform(2.0, 8.0), 1),
        "cr_discharge": round(random.uniform(1.0, 3.5), 1),
        "ultrasound_result": random.choice(["normal", "no hydronephrosis", "bilateral echogenic kidneys"]),
        "location": random.choice(["left MCA territory", "right MCA territory", "posterior circulation", "lacunar"]),
        "tpa_decision": random.choice(["administered", "not administered due to late presentation", "contraindicated"]),
        "bleed_type": random.choice(["upper", "lower"]),
        "hgb_admit": round(random.uniform(5.5, 9.5), 1),
        "procedure": random.choice(["EGD", "colonoscopy"]),
        "finding": random.choice(["gastric ulcer with visible vessel", "colonic AVM", "esophageal varices", "diverticular bleed"]),
        "units": random.randint(2, 6),
        "ph": round(random.uniform(6.9, 7.2), 2),
        "glucose": random.randint(350, 800),
        "complication": complication,
    }
    return template.format(**fills)


def _generate_discharge_note(
    subject_id: int,
    hadm_id: int,
    age: int,
    gender: str,
    condition: str,
    los_days: int,
) -> str:
    """Assemble a realistic discharge summary."""
    gender_str = "male" if gender == "M" else "female"
    pmh = random.sample(PAST_MEDICAL_CONDITIONS, k=random.randint(2, 6))
    meds = random.sample(DISCHARGE_MEDICATIONS, k=random.randint(4, 10))

    note = f"""Admission Date: ___
Discharge Date: ___
Date of Birth: ___
Sex: {gender}
Service: MEDICINE
Allergies: {random.choice(['No Known Allergies', 'Penicillin', 'Sulfa', 'NSAIDs', 'Contrast dye'])}
Attending: ___ MD

Chief Complaint:
{condition.capitalize()}

History of Present Illness:
The patient is a {age}-year-old {gender_str} with a past medical history of {', '.join(pmh[:3])} who presented to the emergency department with {condition}.
The patient reports symptoms began approximately {random.randint(1, 14)} days prior to admission.
{random.choice(['The patient was seen by their primary care physician prior to presentation.', 'The patient called EMS due to worsening symptoms at home.', 'The patient was transferred from an outside hospital for higher level of care.'])}
Review of systems is positive for {random.choice(['fatigue and malaise', 'nausea and vomiting', 'decreased appetite', 'dyspnea on exertion'])}.
Negative for {random.choice(['fever and chills', 'chest pain', 'headache', 'vision changes'])}.

Past Medical History:
{chr(10).join(f'- {c}' for c in pmh)}

Social History:
{random.choice(SOCIAL_HISTORY_TEMPLATES).format(
    age=age, gender=gender_str,
    living_situation=random.choice(LIVING_SITUATIONS),
    tobacco=random.choice(TOBACCO),
    alcohol=random.choice(ALCOHOL),
    drugs=random.choice(DRUGS),
    occupation=random.choice(OCCUPATIONS)
)}

Family History:
{random.choice(['Father with coronary artery disease. Mother with hypertension.', 'Non-contributory.', 'Mother with breast cancer. Father with diabetes.', 'Brother with early MI at age 45.'])}

Physical Exam:
On Admission:
{_generate_vitals()}
{chr(10).join(random.sample(EXAM_FINDINGS, k=random.randint(3, 5)))}

Pertinent Results:
WBC {round(random.uniform(3.5, 22.0), 1)} Hgb {round(random.uniform(7.0, 14.5), 1)} Plt {random.randint(80, 450)}
Na {random.randint(128, 148)} K {round(random.uniform(2.8, 5.8), 1)} Cr {round(random.uniform(0.6, 6.5), 1)} BUN {random.randint(10, 80)}
Troponin: {random.choice(['<0.01 (negative)', f'{round(random.uniform(0.05, 15.0), 2)} (elevated)'])}
BNP: {random.randint(50, 4500)}

Brief Hospital Course:
Patient is a {age}-year-old {gender_str} admitted for {condition}. Hospital course lasted {los_days} days.
{_generate_hospital_course(condition)}

Medications on Admission:
{chr(10).join(f'- {m}' for m in random.sample(DISCHARGE_MEDICATIONS, k=random.randint(2, 6)))}

Discharge Medications:
{chr(10).join(f'- {m}' for m in meds)}

Discharge Disposition:
{random.choice(DISCHARGE_DISPOSITIONS)}

Discharge Diagnosis:
Primary: {condition.capitalize()}
Secondary: {', '.join(pmh[:2])}

Discharge Condition:
{random.choice(['Mental Status: Clear and coherent. Level of Consciousness: Alert and interactive. Activity Status: Ambulatory - Independent.', 'Mental Status: Confused at baseline. Level of Consciousness: Alert and interactive. Activity Status: Requires assistance.', 'Mental Status: Clear and coherent. Level of Consciousness: Alert and interactive. Activity Status: Ambulatory with assistance.'])}

Discharge Instructions:
The patient was instructed to follow up with their primary care physician within {random.randint(3, 14)} days.
The patient was advised regarding medication changes and activity restrictions.
Return precautions discussed.

Followup Instructions:
- Primary care: within {random.randint(3, 7)} days
- Cardiology: within {random.randint(7, 30)} days
- {random.choice(['Nephrology', 'Pulmonology', 'Neurology', 'Gastroenterology'])}: within {random.randint(14, 60)} days
"""
    return note


# ---------------------------------------------------------------------------
# Main data generation functions
# ---------------------------------------------------------------------------

def generate_patients(n: int = 1000) -> pd.DataFrame:
    """Generate synthetic patients table matching MIMIC-IV schema."""
    logger.info("Generating %d synthetic patients...", n)
    subject_ids = list(range(10000000, 10000000 + n))
    genders = np.random.choice(["M", "F"], size=n, p=[0.52, 0.48])
    anchor_ages = np.random.choice(
        range(18, 90),
        size=n,
        p=_age_distribution(range(18, 90)),
    )
    anchor_years = np.random.randint(2100, 2110, size=n)
    anchor_year_groups = [f"{y-2}-{y}" for y in anchor_years]
    dod = [
        (datetime(2100 + random.randint(0, 20), random.randint(1, 12), random.randint(1, 28))).strftime("%Y-%m-%d")
        if random.random() < 0.15 else ""
        for _ in range(n)
    ]

    return pd.DataFrame({
        "subject_id": subject_ids,
        "gender": genders,
        "anchor_age": anchor_ages,
        "anchor_year": anchor_years,
        "anchor_year_group": anchor_year_groups,
        "dod": dod,
    })


def _age_distribution(age_range) -> np.ndarray:
    """Create a realistic age distribution weighted toward older patients (ICU population)."""
    ages = np.array(list(age_range), dtype=float)
    weights = np.where(ages < 40, 0.5, np.where(ages < 65, 1.5, 3.0))
    return weights / weights.sum()


def generate_admissions(patients_df: pd.DataFrame, readmission_rate: float = 0.18) -> pd.DataFrame:
    """Generate synthetic admissions table matching MIMIC-IV schema."""
    logger.info("Generating admissions with %.0f%% readmission rate...", readmission_rate * 100)

    rows = []
    hadm_id = 20000000
    insurance_types = ["Medicare", "Medicaid", "Other"]
    insurance_weights = [0.45, 0.20, 0.35]
    admission_types = ["EMERGENCY", "ELECTIVE", "URGENT", "EW EMER."]
    admission_weights = [0.60, 0.20, 0.10, 0.10]
    languages = ["English", "Spanish", "Portuguese", "Mandarin", "Other"]
    marital_statuses = ["MARRIED", "SINGLE", "WIDOWED", "DIVORCED"]
    races = ["WHITE", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO", "ASIAN", "OTHER"]

    for _, patient in patients_df.iterrows():
        n_admissions = np.random.choice([1, 2, 3, 4], p=[0.55, 0.25, 0.12, 0.08])
        admit_time = _random_date(datetime(2110, 1, 1), datetime(2119, 12, 31))

        for i in range(n_admissions):
            los = max(1, int(np.random.lognormal(mean=1.8, sigma=0.7)))
            disch_time = admit_time + timedelta(days=los)
            expire_flag = 1 if (random.random() < 0.04 and i == n_admissions - 1) else 0

            rows.append({
                "subject_id": patient["subject_id"],
                "hadm_id": hadm_id,
                "admittime": admit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "dischtime": disch_time.strftime("%Y-%m-%d %H:%M:%S"),
                "admission_type": np.random.choice(admission_types, p=admission_weights),
                "admission_location": random.choice(["EMERGENCY ROOM", "PHYSICIAN REFERRAL", "TRANSFER FROM HOSPITAL"]),
                "discharge_location": random.choice(["HOME", "SNF", "REHAB", "EXPIRED", "HOME HEALTH CARE"]),
                "insurance": np.random.choice(insurance_types, p=insurance_weights),
                "language": random.choice(languages),
                "marital_status": random.choice(marital_statuses),
                "race": random.choice(races),
                "edregtime": (admit_time - timedelta(hours=random.randint(1, 8))).strftime("%Y-%m-%d %H:%M:%S"),
                "edouttime": admit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "hospital_expire_flag": expire_flag,
            })
            hadm_id += 1
            # Next admission: ~18% within 30 days to match realistic readmission rate
            if random.random() < 0.18:
                gap_days = random.randint(5, 30)
            else:
                gap_days = random.randint(31, 365)
            admit_time = disch_time + timedelta(days=gap_days)

    admissions_df = pd.DataFrame(rows)

    # Create 30-day readmission label
    admissions_df = _create_readmission_labels(admissions_df)
    logger.info("Admissions generated: %d rows", len(admissions_df))
    return admissions_df


def _create_readmission_labels(admissions_df: pd.DataFrame) -> pd.DataFrame:
    """Add 30-day readmission binary label to admissions DataFrame."""
    admissions_df = admissions_df.copy()
    admissions_df["admittime"] = pd.to_datetime(admissions_df["admittime"])
    admissions_df["dischtime"] = pd.to_datetime(admissions_df["dischtime"])
    admissions_df = admissions_df.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

    readmit_flags = []
    for _, grp in admissions_df.groupby("subject_id"):
        grp = grp.reset_index(drop=True)
        flags = [0] * len(grp)
        for i in range(len(grp) - 1):
            if grp.loc[i, "hospital_expire_flag"] == 1:
                flags[i] = -1  # Exclude — patient died
                continue
            days_to_next = (grp.loc[i + 1, "admittime"] - grp.loc[i, "dischtime"]).days
            flags[i] = 1 if days_to_next <= 30 else 0
        flags[-1] = -1  # Last admission — no follow-up
        readmit_flags.extend(flags)

    admissions_df["readmission_30day"] = readmit_flags
    return admissions_df


def generate_discharge_notes(
    patients_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate synthetic discharge summaries for each admission."""
    logger.info("Generating synthetic discharge notes...")

    conditions = list(HOSPITAL_COURSE_TEMPLATES.keys()) + [
        "chest pain", "syncope", "hypertensive urgency", "altered mental status"
    ]

    rows = []
    note_id = 30000000
    # One note per admission
    admissions_df = admissions_df.merge(
        patients_df[["subject_id", "gender", "anchor_age"]],
        on="subject_id",
        how="left",
    )

    for _, adm in admissions_df.iterrows():
        age = int(adm["anchor_age"])
        gender = adm["gender"]
        los_days = max(1, (pd.to_datetime(adm["dischtime"]) - pd.to_datetime(adm["admittime"])).days)
        condition = random.choice(conditions)
        note_text = _generate_discharge_note(
            subject_id=adm["subject_id"],
            hadm_id=adm["hadm_id"],
            age=age,
            gender=gender,
            condition=condition,
            los_days=los_days,
        )
        rows.append({
            "note_id": note_id,
            "subject_id": adm["subject_id"],
            "hadm_id": adm["hadm_id"],
            "note_type": "Discharge summary",
            "note_seq": 1,
            "charttime": adm["dischtime"],
            "storetime": adm["dischtime"],
            "text": note_text,
        })
        note_id += 1

    notes_df = pd.DataFrame(rows)
    logger.info("Discharge notes generated: %d rows", len(notes_df))
    return notes_df


def run(output_dir: str = "data", n_patients: int = 1000) -> None:
    """Generate all synthetic tables and save to CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    patients_df = generate_patients(n=n_patients)
    admissions_df = generate_admissions(patients_df)
    notes_df = generate_discharge_notes(patients_df, admissions_df)

    patients_df.to_csv(output_path / "synthetic_patients.csv", index=False)
    admissions_df.to_csv(output_path / "synthetic_admissions.csv", index=False)
    notes_df.to_csv(output_path / "synthetic_discharge.csv", index=False)

    # Summary
    valid = admissions_df[admissions_df["readmission_30day"] >= 0]
    readmit_rate = valid["readmission_30day"].mean() * 100
    logger.info("Saved patients (%d), admissions (%d), notes (%d)",
                len(patients_df), len(admissions_df), len(notes_df))
    logger.info("Readmission rate (eligible): %.1f%%", readmit_rate)
    logger.info("Files saved to: %s", output_path.resolve())


if __name__ == "__main__":
    run()
