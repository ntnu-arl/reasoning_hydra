from hydra_to_hov_sg import main
from pathlib import Path
import os
import tqdm

ROOT_PATH = Path("/home/albert/Desktop/hydra_out/3rscan")
COLORMAP_PATH = Path(
    "/home/albert/workspaces/hierarchical_reasoning_ws/src/hydra_ros/hydra_ros/config/color/3rscan.csv"
)

pbar = tqdm.tqdm(ROOT_PATH.iterdir(), desc="Processing directories")
error_list = []
for filepath in pbar:
    print(f"Processing {filepath.name}")
    try:
        main(
            filepath / "backend/copy_dsg_with_mesh.json",
            filepath / "hovsg_format",
            COLORMAP_PATH,
        )
    except Exception as e:
        error_list.append((filepath.name, str(e)))
        print(f"Error processing {filepath.name}: {e}")
    pbar.set_description(f"Processing {filepath.name}")

if len(error_list) > 0:
    print("Errors encountered during processing:")
    for error in error_list:
        print(f"Directory: {error[0]}, Error: {error[1]}")
