## Script to move files from one Google Cloud Storage bucket / project to another

from google.cloud import storage
import pathlib
import pystac
import pystac_client
import os

# from coclicodata.etl.cloud_utils import (
#     dir_to_google_cloud,
#     load_google_credentials,
#     p_drive,
# )

# enable the storage client
client = storage.Client()

# account credentials
# coclico_data_dir = pathlib.Path(p_drive, "11207608-coclico", "FASTTRACK_DATA")
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(coclico_data_dir.joinpath("google_credentials.json"))

# input params
source_project = "bathymetry"
dest_project = "cmems-sdb-11209821-002"
source_bucket_name = "cmems-sdb"
dest_bucket_name = "cmems-isdb"
source_bucket_proj = "intertidal_improved_100m_global"
dest_bucket_proj = "intertidal_improved_100m_global"


# function to move files from one bucket to another
def move_all_files(
    source_project,
    source_bucket_name,
    source_bucket_proj,
    dest_project,
    dest_bucket_name,
    dest_bucket_proj,
):
    # Create storage clients for each project
    source_client = storage.Client(project=source_project)
    destination_client = storage.Client(project=dest_project)

    source_bucket = source_client.bucket(source_bucket_name)
    destination_bucket = destination_client.bucket(dest_bucket_name)

    # List existing files in the destination bucket
    existing_blobs = {
        blob.name
        for blob in destination_client.list_blobs(
            destination_bucket, prefix=dest_bucket_proj
        )
    }

    # List all files under the source prefix
    blobs = list(source_client.list_blobs(source_bucket, prefix=source_bucket_proj))

    for blob in blobs:
        source_path = blob.name  # Full path in the source bucket
        relative_path = source_path[len(source_bucket_proj) :].lstrip(
            "/"
        )  # Remove the prefix to keep relative structure
        destination_path = (
            f"{dest_bucket_proj}/{relative_path}"  # Destination sub-folder
        )

        # Skip if file already exists in the destination bucket
        if destination_path in existing_blobs:
            print(f"Skipping {source_path}, already exists in {dest_bucket_name}.")
            continue

        destination_blob = destination_bucket.blob(destination_path)

        # Copy file to the destination bucket under the new sub-folder
        destination_blob.rewrite(blob)
        print(
            f"Copied {source_path} → {destination_path} in {dest_bucket_name} (Project: {dest_project})"
        )

        # Delete the file from the source after copying (to move instead of copy)
        # blob.delete()
        # print(
        #     f"Deleted {source_path} from {source_bucket_name} (Project: {source_project})"
        # )

    print("All files moved successfully!")


# perform action
move_all_files(
    source_project=source_project,
    source_bucket_name=source_bucket_name,
    source_bucket_proj=source_bucket_proj,
    dest_project=dest_project,
    dest_bucket_name=dest_bucket_name,
    dest_bucket_proj=dest_bucket_proj,
)
