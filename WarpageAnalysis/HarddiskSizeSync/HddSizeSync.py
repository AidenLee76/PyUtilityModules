import os
import psutil
import argparse

# Constants
INITIAL_DUMMY_SIZE_GB = 1

DUMMY_FILE_DIR = 'dummy_files'
DUMMY_FILE_NAME = 'dummy_file'

def create_dummy_file(target_drive, size_mb):
    target_dir = os.path.join(target_drive, DUMMY_FILE_DIR)
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, DUMMY_FILE_NAME)
    with open(file_path, 'wb') as f:
        f.write(b'\0' * (size_mb * 1024 * 1024))

def delete_dummy_file(target_drive):
    file_path = os.path.join(target_drive, DUMMY_FILE_DIR, DUMMY_FILE_NAME)    
    os.remove(file_path)

def get_disk_space(target_drive):    
    disk_usage = psutil.disk_usage(target_drive)
    total_space = round(disk_usage.total / (1024 ** 2) ,1)  # MB
    free_space = round(disk_usage.free / (1024 ** 2), 1)  # MB
    return total_space, free_space

def fill_dummy_space(target_drive, baseline_free_size):    
    file_path = os.path.join(target_drive, DUMMY_FILE_DIR, DUMMY_FILE_NAME)
    if os.path.exists(file_path):
        delete_dummy_file(target_drive)
    
    total_space, free_space = get_disk_space(target_drive)
    if free_space > baseline_free_size:
        required_space = free_space - baseline_free_size
        create_dummy_file(target_drive, int(required_space))        
        
def start_disk_space_management(target_drive):
    
    create_dummy_file(target_drive, INITIAL_DUMMY_SIZE_GB * 1024)  # Create 1GB dummy file
    total_space, free_space = get_disk_space(target_drive)
    
    file_path = os.path.join(target_drive, DUMMY_FILE_DIR, 'disk_space_baseline.txt')

    with open(file_path, 'w') as f:
        f.write(str(free_space))
    print(f"Start: Baseline disk space recorded. free disk size : {round(free_space/1024,2)}GB")

def end_disk_space_management(target_drive):
    file_path = os.path.join(target_drive, DUMMY_FILE_DIR, 'disk_space_baseline.txt')
    with open(file_path, 'r') as f:
        baseline_free_size = float(f.read())
    
        fill_dummy_space(target_drive, baseline_free_size)
                
    print("End: Disk space adjusted to match baseline.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disk Space Management")
    parser.add_argument('action', choices=['start', 'end'], help="Action to perform: start or end")
    args = parser.parse_args()

    if args.action == 'start':
        start_disk_space_management('C:\\')
        start_disk_space_management('D:\\')
    elif args.action == 'end':
        end_disk_space_management('C:\\')
        end_disk_space_management('D:\\')