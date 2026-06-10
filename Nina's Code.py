import polars as pl
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_chisquare

def generate_characteristic_table(input_txt_path, output_csv_path):
    # 1. Load the dataset from your text file using Polars
    print(f"Reading data from {input_txt_path}...")
    df = pl.read_csv(
        input_txt_path,
        separator="\t",
        null_values=["NULL", ""],
        infer_schema_length=0,
        truncate_ragged_lines=True,
    )
    # 2. Filter for Year 2021 and target CPT codes
    # CPT 43775 = Sleeve Gastrectomy, CPT 43644 = Roux-en-Y Gastric Bypass
    df_2021 = df.filter(
        (pl.col("OPYEAR") == "2021") & 
        (pl.col("CPT").cast(pl.Utf8).str.strip_chars().is_in(["43775", "43644"]))
    )

    # Create a clean grouping label column
    df_filtered = df_2021.with_columns(
        pl.when(pl.col("CPT").cast(pl.Utf8).str.strip_chars() == "43775")
        .then(pl.lit("Sleeve"))
        .otherwise(pl.lit("RYGB"))
        .alias("Procedure")
    )

    # Separate the cohorts for baseline analysis
    sleeve_group = df_filtered.filter(pl.col("Procedure") == "Sleeve")
    rygb_group = df_filtered.filter(pl.col("Procedure") == "RYGB")

    # Helper function for continuous metrics (Age, BMI)
    def analyze_continuous(col_name, data_sleeve, data_rygb):
        s_vals = data_sleeve.select(col_name).drop_nulls().to_series()
        r_vals = data_rygb.select(col_name).drop_nulls().to_series()
        
        s_vals = s_vals.cast(pl.Float64).drop_nulls()
        s_arr = s_vals.to_numpy()
        s_mean, s_std = (np.mean(s_arr)), np.std(s_arr) if len(s_arr) > 0 else (0, 0)
        
        r_vals = r_vals.cast(pl.Float64)
        r_arr = r_vals.to_numpy()
        r_arr = r_arr[~np.isnan(r_arr)]
        r_mean, r_std = (np.mean(r_arr)), np.std(r_arr) if len(r_arr) > 0 else (0, 0)
        
        # Calculate p-value via independent T-test
        if len(s_vals) > 1 and len(r_vals) > 1:
            _, p_val = ttest_ind(s_vals, r_vals, equal_var=False)
        else:
            p_val = 1.0
        
        return {
            "Variable": col_name,
            "Sleeve Gastrectomy": f"{s_mean:.2f} (± {s_std:.2f})",
            "Roux-en-Y Gastric Bypass": f"{r_mean:.2f} (± {r_std:.2f})",
            "p-value": f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
        }

    # Helper function for categorical data ratios using statsmodels
    def analyze_categorical(col_name, target_value, data_sleeve, data_rygb):
        s_pos = data_sleeve.filter(pl.col(col_name) == target_value).height
        s_total = data_sleeve.filter(pl.col(col_name).is_not_null()).height
        
        r_pos = data_rygb.filter(pl.col(col_name) == target_value).height
        r_total = data_rygb.filter(pl.col(col_name).is_not_null()).height
        
        s_pct = (s_pos / s_total) * 100 if s_total > 0 else 0
        r_pct = (r_pos / r_total) * 100 if r_total > 0 else 0
        
        count = np.array([s_pos, r_pos])
        nobs = np.array([s_total, r_total])
        
        try:
            if s_total > 0 and r_total > 0:
                _, p_val, _ = proportions_chisquare(count, nobs)
            else:
                p_val = 1.0
        except:
            p_val = np.nan
            
        return {
            "Variable": f"{col_name} ({target_value})",
            "Sleeve Gastrectomy": f"{s_pos} ({s_pct:.1f}%)",
            "Roux-en-Y Gastric Bypass": f"{r_pos} ({r_pct:.1f}%)",
            "p-value": f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
        }

    # Gather rows for final table
    results = []
    
    # Process continuous variables [Colomn Name, Readable Label]
    continuous_features = [
        ("AGE",         "Age (years)"),
        ("BMI",         "Pre-Op BMI"),
        ("BMI_HIGH_BAR","Highest Pre-Op BMI"),
        ("OPLENGTH",    "Operation Length (min)"),
        ("DTDISCH_OP",  "Length of Stay (days)"),
        ("ALBUMIN",     "Albumin (g/dL)"),
        ("CREATININE",  "Creatinine (mg/dL)"),
        ("HCT",         "Hematocrit (%)"),
        ("HEMO",        "HbA1c (%)")
    ]
    for col, label in continuous_features:
        if col in df_filtered.columns:
            row = analyze_continuous(col, sleeve_group, rygb_group)
            row["Variable"] = label
            results.append(row)
            
    # Process categorical variables [Column Name, Variable option]
    cat_features = [
        ("SEX",                          "Female",           "Female Sex"),
        ("DIABETES",                     "Yes, insulin",     "Diabetes (insulin-dependent)"),
        ("DIABETES",                     "Yes, non-insulin", "Diabetes (non-insulin)"),
        ("HYPERTENSION",                 "Yes",              "Hypertension"),
        ("HYPERLIPIDEMIA",               "Yes",              "Hyperlipidemia"),
        ("SLEEP_APNEA",                  "Yes",              "Obstructive Sleep Apnea"),
        ("GERD",                         "Yes",              "GERD"),
        ("SMOKER",                       "Yes",              "Current Smoker"),
        ("COPD",                         "Yes",              "COPD"),
        ("HISTORY_PE",                   "Yes",              "History of PE"),
        ("HISTORY_DVT",                  "Yes",              "History of DVT"),
        ("RENAL_INSUFFICIENCY",          "Yes",              "Renal Insufficiency"),
        ("PREVIOUS_SURGERY",             "Yes",              "Previous Foregut Surgery"),
        ("IMMUNOSUPR_THER",              "Yes",              "Immunosuppressive Therapy"),
        ("THERAPEUTIC_ANTICOAGULATION",  "Yes",              "Therapeutic Anticoagulation"), 
        ("IVC_FILTER",                   "Yes",              "IVC Filter"),
        ("DIALYSIS",                     "Yes",              "On Dialysis"),
        ("ROBOTIC_ASST",                 "Yes",              "Robotic Assist"),
        ("APPROACH_CONVERTED",           "Yes",              "Approach Converted"),
        ("DRAIN_PLACED",                 "Yes",              "Drain Placed"),
        ("REOP30",                       "Yes",              "Reoperation within 30 Days"),
        ("READ30",                       "Yes",              "Readmission within 30 Days"),
        ("INTV30",                       "Yes",              "Intervention within 30 Days"),
        ("POSTOPANASTSLLEAK",            "Yes",              "Anastomotic/Staple Line Leak"),
        ("PULMONARYEMBOLSM",             "Yes",              "Pulmonary Embolism"),
        ("POSTOPSEPSIS",                 "Yes",              "Sepsis"),
        ("POSTOPSEPTICSHOCK",            "Yes",              "Septic Shock"),
        ("VEINTHROMBREQTER",             "Yes",              "Venous Thrombosis"),
        ("TRANSFINTOPPSTOP",             "Yes",              "Blood Transfusion"),
        ("POSTOPPNEUMONIA",              "Yes",              "Pneumonia"), 
        ("UNPLANNEDADMISSIONICU30",      "Yes",              "Unplanned ICU Admission"),
        ("GITRACTBLEED",                 "Yes",              "GI Tract Bleeding"),
        ("BOWELOBSTRUCTION",             "Yes",              "Bowel Obstruction"),
        ("EMERG_VISIT_OUT",              "Yes",              "Emergency Department Visit"),
        ("CVA",                          "Yes",              "Stroke/CVA"),
        ("MYOCARDIALINFR",               "Yes",              "Myocardial Infarction"),
        ("POSTOPUTI",                    "Yes",              "Urinary Tract Infection"),
        ("CDIFF",                        "Yes",              "C. diff Colitis")
    ]
    
    for col, target, label in cat_features:
        if col in df_filtered.columns:
            row = analyze_categorical(col, target, sleeve_group, rygb_group)
            row["Variable"] = label
            results.append(row)

    # 3. Build Table and Save out to CSV
    table_1 = pl.DataFrame(results)
    table_1.write_csv(output_csv_path)
    print(f"Success! Characteristic baseline table exported to: {output_csv_path}")

if __name__ == '__main__':
    # Update these paths with your exact workspace filenames!
    input_file = "/Users/ninaerickson/Downloads/main_2021.txt"
    output_file = "mbsaqip_characteristic_table.csv"
    
    generate_characteristic_table(input_file, output_file)