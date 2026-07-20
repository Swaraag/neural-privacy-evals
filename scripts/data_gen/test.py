from utils import filter_df, populate_bdf_index
from config import DATA_ROOT, PARTICIPANTS_FILE_NAME

filt_df = filter_df(DATA_ROOT / PARTICIPANTS_FILE_NAME)
filt_ids = set(filt_df.index)
bdf_index = populate_bdf_index(DATA_ROOT, filt_ids)

print(f"Subjects after filter_df: {len(filt_ids)}")
print(f"Subjects with BDF files: {len(bdf_index)}")

# Check which subjects are missing BDF
missing_bdf = filt_ids - set(bdf_index.keys())
print(f"Missing BDF: {len(missing_bdf)}")