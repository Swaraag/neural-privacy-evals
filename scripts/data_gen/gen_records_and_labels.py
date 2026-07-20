import json
from utils import filter_df, generate_neural_data, generate_string_neural_data, populate_bdf_index, sanitize
from config import DATA_ROOT, PARTICIPANTS_FILE_NAME
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from universal_utils import get_next_prefix_dir

if __name__ == "__main__":
    participants_file = DATA_ROOT / PARTICIPANTS_FILE_NAME
    
    filt_df = filter_df(participants_file)

    filt_ids = set(filt_df.index)
    bdf_index = populate_bdf_index(DATA_ROOT, filt_ids)

    # generate_string_neural_data(bdf_index) instead if you want the string version
    neural_data = generate_neural_data(bdf_index)

    print("All subjects have been processed.")

    output_dir = DATA_ROOT / "01_attr_inference"
    version_dir = get_next_prefix_dir(output_dir, "version")
    version_dir.mkdir(parents=True, exist_ok=True)

    with open(version_dir / "records.json", "w", encoding="utf-8") as file:
        json.dump(neural_data, file, indent=2)

    labels = {}

    for subj_id in neural_data.keys():
        row = filt_df.loc[subj_id]
        # dumping all information into labels so that the information is available to then determine at experiment time what are the target labels
        labels[subj_id] = {k: sanitize(v) for k, v in row.to_dict().items()}
    with open(version_dir / "labels.json", "w", encoding="utf-8") as file:
        json.dump(labels, file, indent=2)