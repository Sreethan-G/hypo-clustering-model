import pandas as pd

def load_data(path="data/allhypo.data"):
    columns = [
        "age","sex","on_thyroxine","query_on_thyroxine","on_antithyroid_med",
        "sick","pregnant","thyroid_surgery","I131_treatment","query_hypothyroid",
        "query_hyperthyroid","lithium","goitre","tumor","hypopituitary",
        "psych","TSH_measured","TSH","T3_measured","T3","TT4_measured",
        "TT4","T4U_measured","T4U","FTI_measured","FTI","TBG_measured","TBG",
        "referral_source","target"
    ]

    df = pd.read_csv(path, names=columns)
    df = df.replace("?", pd.NA)

    return df