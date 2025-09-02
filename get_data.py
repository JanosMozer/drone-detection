import kagglehub

# Download latest version
path = kagglehub.dataset_download("gray8ed/audio-dataset-of-low-flying-aircraft-aerosonicdb")

print("Path to dataset files:", path)