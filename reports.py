import pandas as pd
from datetime import datetime


def generate_report(data, filename="welfare_report.csv"):
    df = pd.DataFrame(data)
    if "checked_at" not in df.columns:
        df["checked_at"] = datetime.now().isoformat()
    df.to_csv(filename, index=False)
    return filename
