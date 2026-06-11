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
    required_puf_columns = ["AGE", "BMI", "SEX", "OPLENGTH", "READ30", "REOP30", "POSTOPANASTSLLEAK"]
    missing_from_file = [col for col in required_puf_columns if col not in df_2021.columns]
    
    if missing_from_file:
        print(f"\n❌ Error: Missing columns {missing_from_file} in your text file.")
        return

    # 2. Variable Structural Alignment
    print("Mapping variables and building clinical composites...")
    df_filtered = df_2021.with_columns([
        pl.col("AGE").alias("AGEMODEL"),
        pl.col("BMI").alias("BMIMODEL"),
        pl.col("OPLENGTH").alias("OPLENGTH"),
        pl.col("READ30").alias("READMISSION"),
        pl.col("REOP30").alias("REOPERATION"),
        pl.col("POSTOPANASTSLLEAK").alias("LEAK")
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
        ).then(pl.lit("Yes")).otherwise(pl.lit("No")).alias("CARDIACHISTORY"),

        pl.when(
            (pl.col("HISTORY_DVT") == "Yes") |
            (pl.col("THERAPEUTIC_ANTICOAGULATION") == "Yes") |
            (pl.col("VENOUS_STASIS") == "Yes")
        ).then(pl.lit("Yes")).otherwise(pl.lit("No")).alias("VASCULARHISTORY"),
    ])

    # Safe-mapping dictionary for remaining model variables
    mapping_dict = {
        "SEX": "SEX", "HISPANIC": "HISPANIC", "ASACLASS": "ASACLASS", 
        "FUNSTATPRESURG": "FUNSTATPRESURG", "RACE_PUF": "RACEPUF", 
        "DIABETES": "DIABETES", "HYPERTENSION": "HYPERTENSION", "SLEEP_APNEA": "SLEEPAPNEA", 
        "GERD": "GERD", "COPD": "COPD", "RENAL_INSUFFICIENCY": "RENALINSUFFICIENCY", 
        "HYPERLIPIDEMIA": "HYPERLIPIDEMIA", "IMMUNOSUPR_THER": "IMMUNOSUPRTHER", 
        "SMOKER": "SMOKER", "IVC_FILTER": "IVCFILTER", "DIALYSIS": "DIALYSIS", 
        "HISTORY_PE": "HISTORYPE", "ALBUMIN": "ALBUMIN", "CREATININE": "CREATININE", 
        "HCT": "HCT", "HBA1C": "HEMO", "ROBOTIC_ASST": "ROBOTICASST", 
        "DRAIN_PLACED": "DRAINPLACED", "ANASTOMOSIS_CHECKED": "ANASTOMOSISCHECKED", 
        "APPROACH_CONVERTED": "APPROACHCONVERTED", "METH_VTEPROPHYL": "METHVTEPROPHYL"
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
            .then(pl.lit("Yes")).otherwise(pl.lit("No")).alias("SERIOUSCOMPLICATION")
        )
    else:
        df_filtered = df_filtered.with_columns(
            pl.when((pl.col("REOPERATION") == "Yes") | (pl.col("LEAK") == "Yes"))
            .then(pl.lit("Yes")).otherwise(pl.lit("No")).alias("SERIOUSCOMPLICATION")
        )

    df_filtered = df_filtered.with_columns([
        pl.when((pl.col("SERIOUSCOMPLICATION") == "Yes") | (pl.col("READMISSION") == "Yes"))
        .then(pl.lit("Yes")).otherwise(pl.lit("No")).alias("ANYCOMPLICATION")
    ])

    # --- 3. Processing for Regression Framework ---
    print("Formatting variables for multivariable logistic models...")
    # Convert polars -> pandas without relying on pyarrow internals
    # (avoids compatibility issues across polars/pyarrow versions on Python 3.14)
    df_model = pd.DataFrame({col: df_filtered[col].to_list() for col in df_filtered.columns})

    # Scale operative time per 10-minute intervals
    df_model["OPLENGTH10MIN"] = pd.to_numeric(df_model["OPLENGTH"], errors='coerce') / 10.0
    df_model["OPLENGTH10MIN"] = df_model["OPLENGTH10MIN"].fillna(df_model["OPLENGTH10MIN"].median())

    # Map target intraoperative indicators to binary flags
    intraop_flags = ["ROBOTICASST", "DRAINPLACED", "ANASTOMOSISCHECKED", "APPROACHCONVERTED"]
    for col in intraop_flags:
        df_model[col] = df_model[col].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

    # Median Imputation + Missingness Indicators for Laboratories
    impute_metrics = ["ALBUMIN", "CREATININE", "HCT", "HEMO", "AGEMODEL", "BMIMODEL"]
    for var in impute_metrics:
        df_model[var] = pd.to_numeric(df_model[var], errors='coerce')
        df_model[f"{var}missing"] = df_model[var].isna().astype(int)
        df_model[var] = df_model[var].fillna(df_model[var].median())

    # Convert final dependent strings to regression-ready targets (0/1)
    outcomes = ["ANYCOMPLICATION", "SERIOUSCOMPLICATION", "READMISSION", "REOPERATION", "LEAK"]
    for out in outcomes:
        df_model[out] = df_model[out].apply(lambda x: 1 if str(x).strip().lower() in ["yes", "1", "true"] else 0)

    # --- 4. Clean Formula Construction ---
    # FIX: Corrected typo 'C(RACEPU)' to 'C(RACEPUF)'
    preop_pool = [
        ('AGEMODEL', "AGEMODEL"), ('BMIMODEL', "BMIMODEL"), ('C(SEX)', "SEX"), 
        ('C(HISPANIC)', "HISPANIC"), ('C(ASACLASS)', "ASACLASS"), ('C(FUNSTATPRESURG)', "FUNSTATPRESURG"), 
        ('C(RACEPUF)', "RACEPUF"), ('C(DIABETES)', "DIABETES"), ('C(HYPERTENSION)', "HYPERTENSION"), 
        ('C(SLEEPAPNEA)', "SLEEPAPNEA"), ('C(GERD)', "GERD"), ('C(COPD)', "COPD"), 
        ('C(RENALINSUFFICIENCY)', "RENALINSUFFICIENCY"), ('C(HYPERLIPIDEMIA)', "HYPERLIPIDEMIA"), 
        ('C(IMMUNOSUPRTHER)', "IMMUNOSUPRTHER"), ('C(SMOKER)', "SMOKER"), ('C(IVCFILTER)', "IVCFILTER"), 
        ('C(DIALYSIS)', "DIALYSIS"), ('C(CARDIACHISTORY)', "CARDIACHISTORY"), ('C(VASCULARHISTORY)', "VASCULARHISTORY"), 
        ('C(HISTORYPE)', "HISTORYPE"), ('ALBUMIN', "ALBUMIN"), ('ALBUMINmissing', "ALBUMINmissing"), 
        ('CREATININE', "CREATININE"), ('CREATININEmissing', "CREATININEmissing"), ('HCT', "HCT"), 
        ('HCTmissing', "HCTmissing"), ('HEMO', "HEMO"), ('HEMOmissing', "HEMOmissing")
    ]
    
    intraop_pool = [
        ('OPLENGTH10MIN', "OPLENGTH10MIN"), ('C(APPROACHCONVERTED)', "APPROACHCONVERTED"), 
        ('C(METHVTEPROPHYL)', "METHVTEPROPHYL"), ('C(DRAINPLACED)', "DRAINPLACED"), 
        ('ANASTOMOSISCHECKED', "ANASTOMOSISCHECKED"), ('ROBOTICASST', "ROBOTICASST")
    ]

    valid_preop = []
    for formula_str, col_name in preop_pool:
        # Extra safeguard: Prevent crash if a future lookup key has an unexpected mismatch
        if col_name in df_model.columns:
            if df_model[col_name].nunique() > 1:
                if col_name in ["AGEMODEL", "BMIMODEL", "ALBUMIN", "CREATININE", "HCT", "HEMO"]:
                    valid_preop.append(formula_str)
                elif (df_model[col_name] == "Yes").sum() >= 5 or df_model[col_name].dtype != object:
                    valid_preop.append(formula_str)
        else:
            print(f"⚠️ Notice: Column '{col_name}' wasn't found in processed data. Skipping term dynamically.")

    valid_intraop = []
    for formula_str, col_name in intraop_pool:
        if col_name in df_model.columns:
            if df_model[col_name].nunique() > 1:
                valid_intraop.append(formula_str)

    full_model_formula = " + ".join(valid_preop) + " + " + " + ".join(valid_intraop)

    target_factors = {
        'OPLENGTH10MIN': "Operative time (per 10 min)",
        'C(APPROACHCONVERTED)[T.1]': "Approach conversion (MBSAQIP-coded)",
        'C(DRAINPLACED)[T.1]': "Drain placed",
        'ANASTOMOSISCHECKED': "Anastomosis/leak test performed",
        'ROBOTICASST': "Robotic assistance",
        'C(METHVTEPROPHYL)[T.Mechanical only]': "VTE prophylaxis: Mechanical only",
        'C(METHVTEPROPHYL)[T.Missing]': "VTE prophylaxis: Missing",
        'C(METHVTEPROPHYL)[T.Pharmacologic only]': "VTE prophylaxis: Pharmacologic only"
    }

    regression_table = {var: {"Intraoperative Factor": label} for var, label in target_factors.items()}

    # --- 5. Fit Logistic Models Across Endpoints ---
    for outcome in outcomes:
        print(f"Fitting Adjusted Model for outcome: {outcome}...")
        formula = f'{outcome} ~ {full_model_formula}'
        
        try:
            model = smf.logit(formula, data=df_model).fit(method='bfgs', maxiter=400, disp=0)
            coefficients = model.params
            
            try:
                intervals = model.conf_int()
                has_intervals = True
            except:
                has_intervals = False

            for var_name in target_factors.keys():
                if var_name in coefficients.index:
                    or_val = np.exp(coefficients[var_name])
                    
                    if has_intervals and var_name in intervals.index:
                        lower = np.exp(intervals.loc[var_name, 0])
                        upper = np.exp(intervals.loc[var_name, 1])
                        
                        if or_val > 100 or or_val < 0.01 or np.isnan(lower):
                            regression_table[var_name][outcome] = f"{or_val:.2f} (Separated)"
                        else:
                            regression_table[var_name][outcome] = f"{or_val:.2f} ({lower:.2f}-{upper:.2f})"
                    else:
                        regression_table[var_name][outcome] = f"{or_val:.2f} (CI Unstable)"
                else:
                    regression_table[var_name][outcome] = "N/A"
                    
        except Exception as e:
            print(f"⚠️ Skipping statistical compilation for {outcome}: {e}")
            for var_name in target_factors.keys():
                regression_table[var_name][outcome] = "Model Skipped"

    # --- 6. Build Matrix and Save Output ---
    output_rows = list(regression_table.values())
    df_output = pl.DataFrame(output_rows)
    
    rename_dict = {
        "ANYCOMPLICATION": "ANY_COMPLICATION", 
        "SERIOUSCOMPLICATION": "SERIOUS_COMPLICATION",
    }
    
    final_order = ["Intraoperative Factor"]
    for out in outcomes:
        if out in df_output.columns:
            final_order.append(out)
        else:
            df_output = df_output.with_columns(pl.lit("Model Skipped").alias(out))
            final_order.append(out)
    
    # Only rename columns that exist (avoids polars error on missing keys)
    active_renames = {k: v for k, v in rename_dict.items() if k in df_output.columns}
    df_output = df_output.select(final_order).rename(active_renames)
    df_output.write_csv(output_csv_path)
    print(f"\n🎉 Success! The completed Multivariable Regression Table has been successfully written to: {output_csv_path}")


if __name__ == '__main__':
    input_file = "/Users/zoesumner/Desktop/MBSAQIP Data/main_2021.txt"
    output_regression_file = "mbsaqip_regression_table.csv"
    
    generate_regression_table(input_file, output_regression_file)