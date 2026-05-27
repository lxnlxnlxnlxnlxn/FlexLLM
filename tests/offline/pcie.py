import csv
from typing import List, Dict
import subprocess
import argparse


def main(args: argparse.ArgumentParser):
    
    # Parse input arguments for memory range and target device
    start_size = args.start_size        # Starting memory size for the test
    end_size = args.end_size            # Ending memory size for the test
    increment_size = args.increment_size  # Memory size step between tests
    device = args.device                # Target GPU device ID (0 or 1)

    # Command to test Host-to-Device (CPU -> GPU) bandwidth
    htod_command = f"/workspace/data/tools/bandwidthTest --device {device} " \
        f"--memory pageable --mode range --htod " \
        f"--start={start_size} --end={end_size} --increment={increment_size}"
    
    # Command to test Device-to-Host (GPU -> CPU) bandwidth
    dtoh_command = f"/workspace/data/tools/bandwidthTest --device {device} " \
        f"--memory pageable --mode range --dtoh " \
        f"--start={start_size} --end={end_size} --increment={increment_size}"

    try:

        # Run HTOD bandwidth test and capture output
        htod_output = subprocess.check_output(htod_command, shell=True,
            encoding='utf-8', stderr=subprocess.STDOUT)
        print("htod finished !")
        
        # Run DTOH bandwidth test and capture output
        dtoh_output = subprocess.check_output(dtoh_command, shell=True,
            encoding='utf-8', stderr=subprocess.STDOUT)
        print("dtoh finished !")
    
    except subprocess.CalledProcessError as e:
        
        # Raise error if the bandwidth test command fails
        raise RuntimeError(f"command failed: {e} /n {e.output}")
    
    # Parse raw output into structured memory-bandwidth data
    htod_result = get_result(htod_output)
    dtoh_result = get_result(dtoh_output)

    # Save HTOD results to CSV file
    htod_path = "/workspace/data/PCIe/htod.csv"
    
    # Save DTOH results to CSV file
    dtoh_path = "/workspace/data/PCIe/dtoh.csv"
    
    # Write HTOD bandwidth data to CSV
    with open(htod_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['mem', 'bandwidth(MB/ms)']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(htod_result)
    
    # Write DTOH bandwidth data to CSV
    with open(dtoh_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['mem', 'bandwidth(MB/ms)']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dtoh_result)


def get_result(output: str) -> List[Dict]:
    lines = output.strip().split('\n')
    res_list = list()
    
    # Skip header and tail lines; only extract valid test data lines
    for line in lines[9: -4]:
        parts = line.strip().split()
        res_list.append({
            "mem": int(parts[0]),                        # Memory size in bytes
            "bandwidth(MB/ms)": round(float(parts[1]) / 1000, 2)  # Convert MB/s to MB/ms
        })

    return res_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCIe Bandwidth Test Tool for FlexLLM")
    
    parser.add_argument(
        '--start_size', 
        type=int,
        required=True, 
        help="Starting memory size in bytes"
    )
    parser.add_argument(
        '--end_size', 
        type=int, 
        required=True, 
        help="Ending memory size in bytes"
    )
    parser.add_argument(
        '--increment_size', 
        type=int, 
        required=True, 
        help="Increment step size in bytes"
    )
    parser.add_argument(
        '--device', 
        type=int, 
        choices=[0, 1], 
        required=True, 
        help="GPU device ID: 0 or 1"
    )

    args = parser.parse_args()
    main(args=args)
    