def get_next_run_dir(output_dir):
    existing = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    next_num = len(existing) + 1
    return output_dir / f"run_{next_num:03d}"

def get_cur_run_dir(output_dir):
    existing = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    next_num = len(existing)
    return output_dir / f"run_{next_num:03d}"