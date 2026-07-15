def get_next_prefix_dir(output_dir, prefix):
    existing = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith(f"{prefix}_")]
    next_num = len(existing) + 1
    return output_dir / f"{prefix}_{next_num:03d}"

def get_cur_prefix_dir(output_dir, prefix):
    existing = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith(f"{prefix}_")]
    next_num = len(existing)
    return output_dir / f"{prefix}_{next_num:03d}"