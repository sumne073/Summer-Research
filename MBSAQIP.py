import polars as pl
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_chisquare

def generate_characteristic_table(input_txt_path, output_csv_path):
    # 1. Load the dataset from your text file using Polars
    print(f"Reading data from {input_txt_path}...")
    df = pl.read_csv(input_txt_path, separator="\t", infer_schema_length=10000)

    # 2. Filter for Year 2021 and target CPT codes
    # CPT 43775 = Sleeve Gastrectomy, CPT 43644 = Roux-en-Y Gastric Bypass
    df_2021 = df.filter(
        (pl.col("OPYEAR") == 2021) & 
        (pl.col("CPT").cast(pl.Utf8).str.strip_chars().is_in(["43775", "43644"]))
    )

    # Create a clean grouping label column
    df_filtered = df_2021.with_columns(
        pl.when(pl.col("CPT").cast(pl.Utf8).str.strip_chars() == "43775")
        .then(pl.lit("Sleeve"))
        .otherwise(pl.lit("RYGB"))
        .alias("Procedure")
    )
    
    # Create composite columns before splitting into sleeve/rygb groups
    df_filtered = df_filtered.with_columns([
        pl.when(
            (pl.col("MI_ALL_HISTORY") == "Yes") |
            (pl.col("PTC") == "Yes") |
            (pl.col("PCARD") == "Yes")
        ).then(pl.lit("Yes")).otherwise(pl.lit("No")).alias("CARDIAC_HISTORY"),

        pl.when(
            (pl.col("HISTORY_DVT") == "Yes") |
            (pl.col("THERAPEUTIC_ANTICOAGULATION") == "Yes") |
            (pl.col("VENOUS_STASIS") == "Yes")
        ).then(pl.lit("Yes")).otherwise(pl.lit("No")).alias("VASCULAR_HISTORY"),
    ])

    # Separate the cohorts for baseline analysis
    sleeve_group = df_filtered.filter(pl.col("Procedure") == "Sleeve")
    rygb_group = df_filtered.filter(pl.col("Procedure") == "RYGB")
    
    # Helper function for continuous metrics (Age, BMI)
    def analyze_continuous(col_name, data_sleeve, data_rygb):
        s_vals = data_sleeve.select(col_name).drop_nulls().to_series().to_numpy()
        r_vals = data_rygb.select(col_name).drop_nulls().to_series().to_numpy()
        
        s_mean, s_std = np.mean(s_vals), np.std(s_vals) if len(s_vals) > 0 else (0, 0)
        r_mean, r_std = np.mean(r_vals), np.std(r_vals) if len(r_vals) > 0 else (0, 0)
        
        # Calculate p-value via independent T-test
        if len(s_vals) > 1 and len(r_vals) > 1:
            _, p_val = ttest_ind(s_vals, r_vals, equal_var=False)
        else:
            p_val = 1.0
        
        return {
            "Variable": col_name,
            "Sleeve Gastrectomy": f"{s_mean:.2f} (± {s_std:.2f})",
            "Roux-en-Y Gastric Bypass": f"{r_mean:.2f} (± {r_std:.2f})",
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
        }

    # Gather rows for final table
    results = []
    
    # Process continuous variables (Fixed "HEMO" typo here)
    for col in ["AGE", "BMI", "ALBUMIN", "CREATININE", "HCT", "HEMO", "OPLENGTH"]:
        if col in df_filtered.columns:
            results.append(analyze_continuous(col, sleeve_group, rygb_group))
            
    # Process categorical variables [Column Name, Variable option]
    cat_features = [
        # Demographics
        ("SEX", "Female"),
        ("HISPANIC", "Yes"),
        # ASA Class (collapsed to 3 groups per the study)
        ("ASACLASS", "ASA I - Normal/Healthy"),
        ("ASACLASS", "ASA II - Mild systemic disease"),
        ("ASACLASS", "ASA III - Severe systemic disease"),
        ("ASACLASS", "ASA IV - Severe systemic disease threat to life"),
        # Functional status
        ("FUNSTATPRESURG", "Independent"),
        # Race
        ("RACE_PUF", "White"),
        ("RACE_PUF", "Black or African American"),
        ("RACE_PUF", "Unknown/Not Reported"),
        # Comorbidities
        ("DIABETES", "Yes, insulin"),
        ("DIABETES", "Yes non-insulin"),
        ("HYPERTENSION", "Yes"),
        ("SLEEP_APNEA", "Yes"),
        ("GERD", "Yes"),
        ("COPD", "Yes"),
        ("RENAL_INSUFFICIENCY", "Yes"),
        ("HYPERLIPIDEMIA", "Yes"),
        ("IMMUNOSUPR_THER", "Yes"),
        ("SMOKER", "Yes"),
        ("IVC_FILTER", "Yes"),
        ("DIALYSIS", "Yes"),
        # Cardiac history components (for composite)
        ("MI_ALL_HISTORY", "Yes"),
        ("PTC", "Yes"),
        ("PCARD", "Yes"),
        # Vascular history components (for composite)
        ("HISTORY_DVT", "Yes"),
        ("THERAPEUTIC_ANTICOAGULATION", "Yes"),
        ("VENOUS_STASIS", "Yes"),
        ("HISTORY_PE", "Yes"),
        # Surgical / intraoperative
        ("ROBOTIC_ASST", "Yes"),
        ("DRAIN_PLACED", "Yes"),
        ("ANASTOMOSIS_CHECKED", "Yes"),
        ("APPROACH_CONVERTED", "Yes"),
        # VTE prophylaxis method
        ("METH_VTEPROPHYL", "Mechanical and pharmacologic"),
        ("METH_VTEPROPHYL", "Mechanical only"),
        ("METH_VTEPROPHYL", "Pharmacologic only"),
    ]

    for col, target in cat_features:
        if col in df_filtered.columns:
            results.append(analyze_categorical(col, target, sleeve_group, rygb_group))

    # 3. Build Table and Save out to CSV
    table_1 = pl.DataFrame(results)
    table_1.write_csv(output_csv_path)
    print(f"Success! Characteristic baseline table exported to: {output_csv_path}")


if __name__ == '__main__':
    # Update these paths with your exact workspace filenames!
    input_file = "/Users/zoesumner/Desktop/MBSAQIP Data/main_2021.txt"
    output_file = "mbsaqip_characteristic_table.csv"
    
    generate_characteristic_table(input_file, output_file)
