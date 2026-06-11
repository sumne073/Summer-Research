import polars as pl
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def generate_regression_table(input_txt_path, output_csv_path):
    print(f"Reading data from {input_txt_path}...")
    df = pl.read_csv(input_txt_path, separator="\t", infer_schema_length=10000)

    # 1. Standard Cohort Filtering (2021 & Bariatric CPTs)
    df_2021 = df.filter(
        (pl.col("OPYEAR") == 2021) & 
        (pl.col("CPT").cast(pl.Utf8).str.strip_chars().is_in(["43775", "43644"]))
    )

    # --- Schema Validation Stage ---
    required_puf_columns = ["AGE", "BMI", "SEX", "OPLENGTH", "READ30", "REOP30", "COMPL_LEAK"]
    missing_from_file = [col for col in required_puf_columns if col not in df_2021.columns]
    
    if missing_from_file:
        print(f"\n❌ Error: Missing columns {missing_from_file} in your text file.")
        return

    # 2. Variable Structural Alignment
    print("Mapping variables and building clinical composites...")
    df_filtered = df_2021.with_columns([
        pl.col("AGE").alias("AGE_MODEL"),
        pl.col("BMI").alias("BMI_MODEL"),
        pl.col("OPLENGTH").alias("OPLENGTH"),
        pl.col("READ30").alias("READMISSION"),
        pl.col("REOP30").alias("REOPERATION"),
        pl.col("COMPL_LEAK").alias("LEAK")
    ])

    # Procedure type label
    df_filtered = df_filtered.with_columns(
        pl.when(pl.col("CPT").cast(pl.Utf8).str.strip_chars() == "43775")
        .then(pl.lit("Sleeve"))
        .otherwise(pl.lit("RYGB"))
        .alias("Procedure")
    )

    # Re-apply history composites from your characterization framework
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

    # Safe-mapping dictionary for remaining model variables
    mapping_dict = {
        "SEX": "SEX", "HISPANIC": "HISPANIC", "ASACLASS": "ASACLASS", 
        "FUNSTATPRESURG": "FUNSTATPRESURG", "RACE_PUF": "RACE_PUF", 
        "DIABETES": "DIABETES", "HYPERTENSION": "HYPERTENSION", "SLEEP_APNEA": "SLEEP_APNEA", 
        "GERD": "GERD", "COPD": "COPD", "RENAL_INSUFFICIENCY": "RENAL_INSUFFICIENCY", 
        "HYPERLIPIDEMIA": "HYPERLIPIDEMIA", "IMMUNOSUPR_THER": "IMMUNOSUPR_THER", 
        "SMOKER": "SMOKER", "IVC_FILTER": "IVC_FILTER", "DIALYSIS": "DIALYSIS", 
        "HISTORY_PE": "HISTORY_PE", "ALBUMIN": "ALBUMIN", "CREATININE": "CREATININE", 
        "HCT": "HCT", "HBA1C": "HEMO", "ROBOTIC_ASST": "ROBOTIC_ASST", 
        "DRAIN_PLACED": "DRAIN_PLACED", "ANASTOMOSIS_CHECKED": "ANASTOMOSIS_CHECKED", 
        "APPROACH_CONVERTED": "APPROACH_CONVERTED", "METH_VTEPROPHYL": "METH_VTEPROPHYL"
    }
    
    for raw_name, model_name in mapping_dict.items():
        if raw_name in df_filtered.columns:
            df_filtered = df_filtered.with_columns(pl.col(raw_name).alias(model_name))
        else:
            df_filtered = df_filtered.with_columns(pl.lit("No").alias(model_name))

    # Construct Composite Complication Endpoints
    mortality_col = "PUF_MORTALITY" if "PUF_MORTALITY" in df_filtered.columns else ("DEATH30" if "DEATH30" in df_filtered.columns else None)
    
    if mortality_col:
        df_filtered = df_filtered.with_columns(
            pl.when((pl.col("REOPERATION") == "Yes") | (pl.col("LEAK") == "Yes") | (pl.col(mortality_col) == "Yes"))
            .then(pl.lit("Yes")).otherwise(pl.lit("No")).alias("SERIOUS_COMPLICATION")
        )
    else:
        df_filtered = df_filtered.with_columns(
            pl.when((pl.col("REOPERATION") == "Yes") | (pl.col("LEAK") == "Yes"))
            .then(pl.lit("Yes")).otherwise(pl.lit("No")).alias("SERIOUS_COMPLICATION")
        )

    df_filtered = df_filtered.with_columns([
        pl.when((pl.col("SERIOUS_COMPLICATION") == "Yes") | (pl.col("READMISSION") == "Yes"))
        .then(pl.lit("Yes")).otherwise(pl.lit("No")).alias("ANY_COMPLICATION")
    ])

    # --- 3. Processing for Regression Framework ---
    print("Formatting variables for multivariable logistic models...")
    df_model = df_filtered.to_pandas()

    # Scale operative time per 10-minute intervals
    df_model["OPLENGTH_10MIN"] = pd.to_numeric(df_model["OPLENGTH"], errors='coerce') / 10.0
    df_model["OPLENGTH_10MIN"] = df_model["OPLENGTH_10MIN"].fillna(df_model["OPLENGTH_10MIN"].median())

    # Map target intraoperative indicators to binary flags
    intraop_flags = ["ROBOTIC_ASST", "DRAIN_PLACED", "ANASTOMOSIS_CHECKED", "APPROACH_CONVERTED"]
    for col in intraop_flags:
        df_model[col] = df_model[col].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

    # Median Imputation + Missingness Indicators for Laboratories
    impute_metrics = ["ALBUMIN", "CREATININE", "HCT", "HEMO", "AGE_MODEL", "BMI_MODEL"]
    for var in impute_metrics:
        df_model[var] = pd.to_numeric(df_model[var], errors='coerce')
        df_model[f"{var}_missing"] = df_model[var].isna().astype(int)
        df_model[var] = df_model[var].fillna(df_model[var].median())

    # Convert final dependent strings to regression-ready targets (0/1)
    outcomes = ["ANY_COMPLICATION", "SERIOUS_COMPLICATION", "READMISSION", "REOPERATION", "LEAK"]
    for out in outcomes:
        df_model[out] = df_model[out].apply(lambda x: 1 if str(x).strip().lower() in ["yes", "1", "true"] else 0)

    # --- 4. Dynamic Variance Screening Stage ---
    print("Screening variables for zero-variance components...")
    preop_pool = [
        ('Q("AGE_MODEL")', "AGE_MODEL"), ('Q("BMI_MODEL")', "BMI_MODEL"), ('C(Q("SEX"))', "SEX"), 
        ('C(Q("HISPANIC"))', "HISPANIC"), ('C(Q("ASACLASS"))', "ASACLASS"), ('C(Q("FUNSTATPRESURG"))', "FUNSTATPRESURG"), 
        ('C(Q("RACE_PUF"))', "RACE_PUF"), ('C(Q("DIABETES"))', "DIABETES"), ('C(Q("HYPERTENSION"))', "HYPERTENSION"), 
        ('C(Q("SLEEP_APNEA"))', "SLEEP_APNEA"), ('C(Q("GERD"))', "GERD"), ('C(Q("COPD"))', "COPD"), 
        ('C(Q("RENAL_INSUFFICIENCY"))', "RENAL_INSUFFICIENCY"), ('C(Q("HYPERLIPIDEMIA"))', "HYPERLIPIDEMIA"), 
        ('C(Q("IMMUNOSUPR_THER"))', "IMMUNOSUPR_THER"), ('C(Q("SMOKER"))', "SMOKER"), ('C(Q("IVC_FILTER"))', "IVC_FILTER"), 
        ('C(Q("DIALYSIS"))', "DIALYSIS"), ('C(Q("CARDIAC_HISTORY"))', "CARDIAC_HISTORY"), ('C(Q("VASCULAR_HISTORY"))', "VASCULAR_HISTORY"), 
        ('C(Q("HISTORY_PE"))', "HISTORY_PE"), ('Q("ALBUMIN")', "ALBUMIN"), ('Q("ALBUMIN_missing")', "ALBUMIN_missing"), 
        ('Q("CREATININE")', "CREATININE"), ('Q("CREATININE_missing")', "CREATININE_missing"), ('Q("HCT")', "HCT"), 
        ('Q("HCT_missing")', "HCT_missing"), ('Q("HEMO")', "HEMO"), ('Q("HEMO_missing")', "HEMO_missing")
    ]
    
    intraop_pool = [
        ('Q("OPLENGTH_10MIN")', "OPLENGTH_10MIN"), ('C(Q("APPROACH_CONVERTED"))', "APPROACH_CONVERTED"), 
        ('C(Q("METH_VTEPROPHYL"))', "METH_VTEPROPHYL"), ('C(Q("DRAIN_PLACED"))', "DRAIN_PLACED"), 
        ('Q("ANASTOMOSIS_CHECKED")', "ANASTOMOSIS_CHECKED"), ('Q("ROBOTIC_ASST")', "ROBOTIC_ASST")
    ]

    valid_preop = []
    for formula_str, col_name in preop_pool:
        if df_model[col_name].nunique() > 1:
            valid_preop.append(formula_str)

    valid_intraop = []
    for formula_str, col_name in intraop_pool:
        if df_model[col_name].nunique() > 1:
            valid_intraop.append(formula_str)

    full_model_formula = " + ".join(valid_preop) + " + " + " + ".join(valid_intraop)

    target_factors = {
        'Q("OPLENGTH_10MIN")': "Operative time (per 10 min)",
        'C(Q("APPROACH_CONVERTED"))[T.1]': "Approach conversion (MBSAQIP-coded)",
        'C(Q("DRAIN_PLACED"))[T.1]': "Drain placed",
        'Q("ANASTOMOSIS_CHECKED")': "Anastomosis/leak test performed",
        'Q("ROBOTIC_ASST")': "Robotic assistance",
        'C(Q("METH_VTEPROPHYL"))[T.Mechanical only]': "VTE prophylaxis: Mechanical only",
        'C(Q("METH_VTEPROPHYL"))[T.Missing]': "VTE prophylaxis: Missing",
        'C(Q("METH_VTEPROPHYL"))[T.Pharmacologic only]': "VTE prophylaxis: Pharmacologic only"
    }

    regression_table = {var: {"Intraoperative Factor": label} for var, label in target_factors.items()}

    # --- 5. Fit Logistic Models Across Endpoints ---
    for outcome in outcomes:
        print(f"Fitting Adjusted Model for outcome: {outcome}...")
        formula = f'Q("{outcome}") ~ {full_model_formula}'
        
        try:
            # FIX: Switched optimizer method to 'bfgs' with maxiter=400 to handle data separation gracefully
            model = smf.logit(formula, data=df_model).fit(method='bfgs', maxiter=400, disp=0)
            coefficients = model.params
            intervals = model.conf_int()
            
            for var_name in target_factors.keys():
                if var_name in coefficients.index:
                    or_val = np.exp(coefficients[var_name])
                    lower = np.exp(intervals.loc[var_name, 0])
                    upper = np.exp(intervals.loc[var_name, 1])
                    
                    # Catch and format separated extreme/infinite values neatly
                    if or_val > 100 or or_val < 0.01:
                        regression_table[var_name][outcome] = "Inf (Separated)"
                    else:
                        regression_table[var_name][outcome] = f"{or_val:.2f} ({lower:.2f}-{upper:.2f})"
                else:
                    regression_table[var_name][outcome] = "N/A"
        except Exception as e:
            print(f"⚠️ Stability warning for endpoint {outcome}: {e}")
            for var_name in target_factors.keys():
                regression_table[var_name][outcome] = "Data Unstable"

    # --- 6. Build Matrix and Save Output ---
    output_rows = list(regression_table.values())
    df_output = pl.DataFrame(output_rows)
    final_order = ["Intraoperative Factor"] + outcomes
    df_output = df_output.select(final_order)
    
    df_output.write_csv(output_csv_path)
    print(f"\n🎉 Success! The completed Multivariable Regression Table has been generated at: {output_csv_path}")


if __name__ == '__main__':
    input_file = "/Users/zoesumner/Desktop/MBSAQIP Data/main_2021.txt"
    output_regression_file = "mbsaqip_regression_table.csv"
    
    generate_regression_table(input_file, output_regression_file)