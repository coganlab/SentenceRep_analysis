import os
import shutil

# Define the source and destination directories
src_dir = r"C:\Users\Jakda\Box\CoganLab\BIDS-1.3_SentenceOld"
dst_dir = r"C:\Users\Jakda\Box\CoganLab\BIDS-1.4_SentenceRep"

for dirpath, dirnames, filenames in os.walk(src_dir):
    for filename in filenames:
        # Check if the filename contains the keyword "channels"
        if "channels" in filename:
            # Construct full file path
            src_file = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(dirpath, src_dir)
            dst_file = os.path.join(dst_dir, relative_path, filename)

            # Check if the destination directory exists
            if os.path.exists(os.path.dirname(dst_file)):
                # Copy the file to the destination directory, overwriting any existing file
                shutil.copy2(src_file, dst_file)
