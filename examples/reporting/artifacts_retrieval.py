# ClearML - example code, retrieve other task artifacts and print the artifacts
# Please run examples/reporting/artifacts.py example before running this example
#
from pprint import pprint

from clearml import Task


def main():
    # Getting the task we want to get the artifacts from
    artifacts_task = Task.get_task(
        project_name='examples',
        task_name='Artifacts example',
        task_filter={'status': ['completed']}
    )

    # getting the numpy object back
    numpy_artifact = artifacts_task.artifacts['Numpy Eye'].get()
    print(f"numpy_artifact is:\n{numpy_artifact}\n")

    # download the numpy object as a npz file
    download_numpy_artifact = artifacts_task.artifacts['Numpy Eye'].get_local_copy()
    print(f"download_numpy_artifact path is:\n{download_numpy_artifact}\n")

    # getting the PIL Image object
    pil_artifact = artifacts_task.artifacts['pillow_image'].get()
    print(f"pil_artifact is:\n{pil_artifact}\n")

    # getting the pandas object
    pandas_artifact = artifacts_task.artifacts['Pandas'].get()
    print(f"pandas_artifact is:\n{pandas_artifact}\n")

    # getting the dictionary object
    dictionary_artifact = artifacts_task.artifacts['dictionary'].get()
    print("dictionary_artifact is:\n")
    pprint(dictionary_artifact)

    # getting the train DataFrame
    df_artifact = artifacts_task.artifacts['train'].get()
    print(f"df_artifact is:\n{df_artifact}\n")

    # download the train DataFrame csv in the same format as in the UI (gz file)
    df_artifact_as_gz = artifacts_task.artifacts['train'].get_local_copy()
    print(f"df_artifact_as_gz path is:\n{df_artifact_as_gz}\n")

    # download the wildcard jpegs images (getting the zip file already extracted into a cached folder),
    # the path containing those will be returned
    jpegs_artifact = artifacts_task.artifacts['wildcard jpegs'].get()
    print(f"jpegs_artifact path is:\n{jpegs_artifact}\n")

    # download the local folder that was uploaded (getting the zip file already extracted into a cached folder),
    # the path containing those will be returned
    local_folder_artifact = artifacts_task.artifacts['local folder'].get()
    print(f"local_folder_artifact path is:\n{local_folder_artifact}\n")

    # download the local folder that was uploaded (getting the zip file without extracting it),
    # the path containing the zip file will be returned
    local_folder_artifact_as_zip = artifacts_task.artifacts['local folder'].get_local_copy(extract_archive=False)
    print(f"local_folder_artifact_as_zip path is:\n{local_folder_artifact_as_zip}\n")

    # download the local file that was uploaded (getting the zip file already extracted into a cached folder),
    # the path containing this file will be returned
    local_file_artifact = artifacts_task.artifacts['local file'].get()
    print(f"local_file_artifact path is:\n{local_file_artifact}\n")


if __name__ == '__main__':
    main()
