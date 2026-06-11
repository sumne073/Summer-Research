import polars as pl
import numpy as np


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def summarize_continuous(df_s, df_r, col, label):
    s_vals = df_s.select(col).drop_nulls().to_series().cast(pl.Float64).to_numpy()
    r_vals = df_r.select(col).drop_nulls().to_series().cast(pl.Float64).to_numpy()
    s_mean = np.mean(s_vals) if len(s_vals) else np.nan
    s_sd   = np.std(s_vals)  if len(s_vals) else np.nan
    r_mean = np.mean(r_vals) if len(r_vals) else np.nan
    r_sd   = np.std(r_vals)  if len(r_vals) else np.nan
    return {
        "Variable": label,
        "Sleeve Gastrectomy": f"{s_mean:.2f} ± {s_sd:.2f}",
        "Roux-en-Y Gastric Bypass": f"{r_mean:.2f} ± {r_sd:.2f}"
    }


def summarize_binary(df_s, df_r, col, positive_values, label):
    s_pos   = df_s.filter(pl.col(col).cast(pl.Utf8).str.strip_chars().is_in([str(v) for v in positive_values])).height
    r_pos   = df_r.filter(pl.col(col).cast(pl.Utf8).str.strip_chars().is_in([str(v) for v in positive_values])).height
    s_total = df_s.filter(pl.col(col).is_not_null()).height
    r_total = df_r.filter(pl.col(col).is_not_null()).height
    s_pct = (s_pos / s_total * 100) if s_total else 0
    r_pct = (r_pos / r_total * 100) if r_total else 0
    return {
        "Variable": label,
        "Sleeve Gastrectomy": f"{s_pos} ({s_pct:.1f}%)",
        "Roux-en-Y Gastric Bypass": f"{r_pos} ({r_pct:.1f}%)"
    }


def generate_characteristic_table(input_txt, output_csv):

    print(f"Reading data from {input_txt}...")

    df = (
        pl.scan_csv(input_txt, separator="\t", infer_schema_length=0, truncate_ragged_lines=True)
        .filter(
            (pl.col("OPYEAR").cast(pl.String).str.strip_chars() == "2021") &
            (pl.col("CPT").cast(pl.String).str.strip_chars().is_in(["43775", "43644"]))
        )
        .collect()
    )

    print(f"Loaded {df.shape[0]} rows")

    df = df.with_columns(
        pl.when(pl.col("CPT").cast(pl.Utf8).str.strip_chars() == "43775")
        .then(pl.lit("Sleeve"))
        .otherwise(pl.lit("RYGB"))
        .alias("Procedure")
    )

    sleeve = df.filter(pl.col("Procedure") == "Sleeve")
    rygb   = df.filter(pl.col("Procedure") == "RYGB")

    results = []

    results.append({
        "Variable": "Total N",
        "Sleeve Gastrectomy": str(sleeve.height),
        "Roux-en-Y Gastric Bypass": str(rygb.height)
    })

    # ── Continuous variables ──────────────────────────────────────────────────

    continuous_map = {
        "Age (years)":            ["AGE"],
        "Pre-op BMI":             ["BMI"],
        "Highest pre-op BMI":     ["HIGHEST_BMI", "BMI_HIGHEST"],
        "Operation length (min)": ["OPTIME", "OPERATIVE_TIME"],
        "Length of stay (days)":  ["LOS"],
        "Albumin (g/dL)":         ["ALBUMIN"],
        "Creatinine (mg/dL)":     ["CREATININE"],
        "Hematocrit (%)":         ["HCT", "HEMATOCRIT"],
        "HbA1c (%)":              ["HBA1C", "A1C"],
    }

    for label, names in continuous_map.items():
        col = find_column(df, names)
        if col:
            results.append(summarize_continuous(sleeve, rygb, col, label))

    # ── Binary variables ──────────────────────────────────────────────────────

    positive = ["Yes", "YES", "Y", "1", "Insulin"]

    binary_map = {
        "Female sex":                    ["SEX"],
        "Hypertension":                  ["HYPERTENSION"],
        "Hyperlipidemia":                ["HYPERLIPIDEMIA"],
        "Obstructive sleep apnea":       ["SLEEPAPNEA"],
        "GERD":                          ["GERD"],
        "Current smoker":                ["SMOKER"],
        "COPD":                          ["COPD"],
        "History of PE":                 ["HISTORY_PE"],
        "History of DVT":                ["HISTORY_DVT"],
        "Renal insufficiency":           ["RENAL_INSUFFICIENCY"],
        "Previous foregut surgery":      ["FOREGUT_SURGERY"],
        "Immunosuppressive therapy":     ["IMMUNOSUPPRESSIVE"],
        "Therapeutic anticoagulation":   ["THERAPEUTIC_ANTICOAG"],
        "IVC filter":                    ["IVC_FILTER"],
        "On dialysis":                   ["DIALYSIS"],
        "Robotic assist":                ["ROBOTIC_ASSIST"],
        "Approach converted":            ["CONVERSION"],
        "Drain placed":                  ["DRAIN_PLACED"],
        "Reoperation within 30 days":    ["REOPERATION"],
        "Readmission within 30 days":    ["READMISSION"],
        "Intervention within 30 days":   ["INTERVENTION"],
        "Anastomotic/staple line leak":  ["LEAK"],
        "Pulmonary embolism":            ["PE"],
        "Sepsis":                        ["SEPSIS"],
        "Septic shock":                  ["SEPTIC_SHOCK"],
        "Venous thrombosis":             ["VTE"],
        "Blood transfusion":             ["TRANSFUSION"],
        "Pneumonia":                     ["PNEUMONIA"],
        "Unplanned ICU admission":       ["UNPLANNED_ICU"],
        "GI tract bleeding":             ["GI_BLEED"],
        "Bowel obstruction":             ["BOWEL_OBSTRUCTION"],
        "Emergency department visit":    ["ED_VISIT"],
        "Stroke/CVA":                    ["STROKE"],
        "Myocardial infarction":         ["MI"],
        "Urinary tract infection":       ["UTI"],
        "C. diff colitis":               ["CDIFF"],
    }

    for label, names in binary_map.items():
        col = find_column(df, names)
        if col:
            results.append(summarize_binary(sleeve, rygb, col, positive, label))

    # ── Diabetes special handling ─────────────────────────────────────────────

    diabetes_col = find_column(df, ["DIABETES"])
    if diabetes_col:
        results.append(summarize_binary(sleeve, rygb, diabetes_col, ["Insulin"], "Diabetes (insulin-dependent)"))
        results.append(summarize_binary(sleeve, rygb, diabetes_col, ["Yes non-insulin"], "Diabetes (non-insulin)"))

    # ── Export ────────────────────────────────────────────────────────────────

    results_str = [{k: str(v) for k, v in row.items()} for row in results]
    table = pl.DataFrame(results_str)
    table.write_csv(output_csv)
    print(f"Table exported to {output_csv}")


if __name__ == "__main__":
    generate_characteristic_table(
        "/Users/zoyahasan/Desktop/main_2021.txt",
        "mbsaqip_characteristics_table.csv"
    )