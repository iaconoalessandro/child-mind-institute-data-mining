# Feature descriptions for `cmi_internet.csv`

This file documents each column present in `cmi_internet.csv` using the meanings from `data_dictionary.csv`.



## Fields

- `Basic_Demos-Enroll_Season` (str)
  - Description: Season of enrollment.
  - Values: Spring, Summer, Fall, Winter

- `Basic_Demos-Age` (float)
  - Description: Age of participant

- `Basic_Demos-Sex` (categorical int)
  - Description: Sex of participant
  - Values: 0, 1
  - Value labels: 0=Male, 1=Female

- `CGAS-Season` (str)
  - Description: Season of participation (Children's Global Assessment Scale)
  - Values: Spring, Summer, Fall, Winter

- `CGAS-CGAS_Score` (int)
  - Description: Children's Global Assessment Scale Score

- `Physical-Season` (str)
  - Description: Season of participation (Physical Measures)
  - Values: Spring, Summer, Fall, Winter

- `Physical-BMI` (float)
  - Description: Body Mass Index (kg/m^2)

- `Physical-Height` (float)
  - Description: Height (inches)

- `Physical-Weight` (float)
  - Description: Weight (lbs)

- `Physical-Waist_Circumference` (int)
  - Description: Waist circumference (in)

- `Physical-Diastolic_BP` (int)
  - Description: Diastolic blood pressure (mmHg)

- `Physical-HeartRate` (int)
  - Description: Heart rate (beats/min)

- `Physical-Systolic_BP` (int)
  - Description: Systolic blood pressure (mmHg)

- `Fitness_Endurance-Season` (str)
  - Description: Season of participation (FitnessGram endurance)
  - Values: Spring, Summer, Fall, Winter

- `Fitness_Endurance-Max_Stage` (int)
  - Description: Maximum treadmill stage reached

- `Fitness_Endurance-Time_Mins` (int)
  - Description: Minutes component of treadmill time completed

- `Fitness_Endurance-Time_Sec` (int)
  - Description: Seconds component of treadmill time completed

- `FGC-Season` (str)
  - Description: Season of participation (FitnessGram child)
  - Values: Spring, Summer, Fall, Winter

- `FGC-FGC_CU` (int)
  - Description: Curl-up total

- `FGC-FGC_CU_Zone` (categorical int)
  - Description: Curl up fitness zone
  - Values: 0, 1
  - Labels: 0=Needs Improvement, 1=Healthy Fitness Zone

- `FGC-FGC_GSND` (float)
  - Description: Grip strength total (non-dominant)

- `FGC-FGC_GSND_Zone` (categorical int)
  - Description: Grip strength fitness zone (non-dominant)
  - Values: 1,2,3
  - Labels: 1=Weak, 2=Normal, 3=Strong

- `FGC-FGC_GSD` (float)
  - Description: Grip strength total (dominant)

- `FGC-FGC_GSD_Zone` (categorical int)
  - Description: Grip strength fitness zone (dominant)
  - Values: 1,2,3
  - Labels: 1=Weak, 2=Normal, 3=Strong

- `FGC-FGC_PU` (int)
  - Description: Push-up total

- `FGC-FGC_PU_Zone` (categorical int)
  - Description: Push-up fitness zone
  - Values: 0,1
  - Labels: 0=Needs Improvement, 1=Healthy Fitness Zone

- `FGC-FGC_SRL` (float)
  - Description: Sit & Reach total (left side)

- `FGC-FGC_SRL_Zone` (categorical int)
  - Description: Sit & Reach fitness zone (left side)
  - Values: 0,1
  - Labels: 0=Needs Improvement, 1=Healthy Fitness Zone

- `FGC-FGC_SRR` (float)
  - Description: Sit & Reach total (right side)

- `FGC-FGC_SRR_Zone` (categorical int)
  - Description: Sit & Reach fitness zone (right side)
  - Values: 0,1
  - Labels: 0=Needs Improvement, 1=Healthy Fitness Zone

- `FGC-FGC_TL` (int)
  - Description: Trunk lift total

- `FGC-FGC_TL_Zone` (categorical int)
  - Description: Trunk lift fitness zone
  - Values: 0,1
  - Labels: 0=Needs Improvement, 1=Healthy Fitness Zone

- `BIA-Season` (str)
  - Description: Season of participation (Bio-electric Impedance Analysis)
  - Values: Spring, Summer, Fall, Winter

- `BIA-BIA_Activity_Level_num` (categorical int)
  - Description: Activity Level
  - Values: 1,2,3,4,5
  - Labels: 1=Very Light, 2=Light, 3=Moderate, 4=Heavy, 5=Exceptional

- `BIA-BIA_BMC` (float)
  - Description: Bone Mineral Content

- `BIA-BIA_BMI` (float)
  - Description: Body Mass Index (from BIA)

- `BIA-BIA_BMR` (float)
  - Description: Basal Metabolic Rate

- `BIA-BIA_DEE` (float)
  - Description: Daily Energy Expenditure

- `BIA-BIA_ECW` (float)
  - Description: Extracellular Water

- `BIA-BIA_FFM` (float)
  - Description: Fat Free Mass

- `BIA-BIA_FFMI` (float)
  - Description: Fat Free Mass Index

- `BIA-BIA_FMI` (float)
  - Description: Fat Mass Index

- `BIA-BIA_Fat` (float)
  - Description: Body Fat Percentage

- `BIA-BIA_Frame_num` (categorical int)
  - Description: Body Frame
  - Values: 1,2,3
  - Labels: 1=Small, 2=Medium, 3=Large

- `BIA-BIA_ICW` (float)
  - Description: Intracellular Water

- `BIA-BIA_LDM` (float)
  - Description: Lean Dry Mass

- `BIA-BIA_LST` (float)
  - Description: Lean Soft Tissue

- `BIA-BIA_SMM` (float)
  - Description: Skeletal Muscle Mass

- `BIA-BIA_TBW` (float)
  - Description: Total Body Water

- `PAQ_A-Season` (str)
  - Description: Season of participation (Physical Activity Questionnaire - Adolescents)

- `PAQ_A-PAQ_A_Total` (float)
  - Description: Activity Summary Score (Adolescents)

- `PAQ_C-Season` (str)
  - Description: Season of participation (Physical Activity Questionnaire - Children)

- `PAQ_C-PAQ_C_Total` (float)
  - Description: Activity Summary Score (Children)

- `PCIAT-Season` (str)
  - Description: Season of participation (Parent-Child Internet Addiction Test)

- `PCIAT-PCIAT_01` to `PCIAT-PCIAT_20` (categorical int)
  - Description: 20 item Likert-style responses measuring behaviors related to internet use (e.g., disobeying time limits, neglecting chores, mood when offline).
  - Values: 0,1,2,3,4,5
  - Labels: 0=Does Not Apply, 1=Rarely, 2=Occasionally, 3=Frequently, 4=Often, 5=Always

- `PCIAT-PCIAT_Total` (int)
  - Description: Total score for PCIAT
  - Notes: Severity/Impairment Index provided in dictionary: 0-30=None; 31-49=Mild; 50-79=Moderate; 80-100=Severe.

- `SDS-Season` (str)
  - Description: Season of participation (Sleep Disturbance Scale)

- `SDS-SDS_Total_Raw` (int)
  - Description: Total raw score for Sleep Disturbance Scale

- `SDS-SDS_Total_T` (int)
  - Description: Total T-score for Sleep Disturbance Scale

- `PreInt_EduHx-Season` (str)
  - Description: Season of participation (Internet Use / Pre-intake education history)

- `PreInt_EduHx-computerinternet_hoursday` (categorical int)
  - Description: Hours of using computer/internet
  - Values: 0,1,2,3
  - Labels: 0=Less than 1h/day, 1=Around 1h/day, 2=Around 2hs/day, 3=More than 3hs/day


---

## Additional notes and recommendations

- The authoritative meanings are taken from `data_dictionary.csv`. If you need full programmatic verification (e.g., detect additional columns present in `cmi_internet.csv` that are not in the dictionary), I can run a script to list all dataset columns and flag unmatched ones.

- For categorical fields with numeric encodings (e.g., `Basic_Demos-Sex`, `FGC-*_*_Zone`, `BIA-BIA_Activity_Level_num`, PCIAT items), treat them as ordered or nominal as appropriate for analysis and document choices in reports.

- Normalize or standardize continuous measures (e.g., `Physical-BMI`, BIA measures) before distance-based methods.


