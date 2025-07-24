from pathlib import Path


class CFG:
    root = Path(__file__).resolve().parent.parent
    env_file = root.joinpath('.env')

    data = root.joinpath('data')
    table_description = data.joinpath("table_description 1.xlsx")
    golden_dataset = data.joinpath("golden_dataset_export_20250716_095952 1.csv")
