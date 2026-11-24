import kagglehub

# Download latest version
path = kagglehub.dataset_download("iimmortall/s2r-hdr")

print("Path to dataset files:", path)