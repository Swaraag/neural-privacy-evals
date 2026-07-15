import json
from utils import filter_df, generate_neural_data, populate_bdf_index, sanitize
from config import DATA_ROOT, PARTICIPANTS_FILE_NAME

if __name__ == "__main__":
    participants_file = DATA_ROOT / PARTICIPANTS_FILE_NAME
    
    filt_df = filter_df(participants_file)

    filt_ids = set(filt_df.index)
    bdf_index = populate_bdf_index(DATA_ROOT, filt_ids)

    neural_data = generate_neural_data(bdf_index)

    (DATA_ROOT / "01_attr_inference").mkdir(exist_ok=True)
    with open(DATA_ROOT / "01_attr_inference" / "records.json", "w", encoding="utf-8") as file:
        json.dump(neural_data, file, indent=2)

    labels = {}

    for subj_id in neural_data.keys():
        row = filt_df.loc[subj_id]
        # dumping all information into labels so that the information is available to then determine at experiment time what are the target labels
        labels[subj_id] = {k: sanitize(v) for k, v in row.to_dict().items()}
    with open(DATA_ROOT / "01_attr_inference_raw" / "labels.json", "w", encoding="utf-8") as file:
        json.dump(labels, file, indent=2)