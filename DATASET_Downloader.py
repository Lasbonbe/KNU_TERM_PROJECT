import kagglehub

# Download latest version
path = kagglehub.dataset_download("nzigulic/military-equipment")

print("Path to dataset files:", path)