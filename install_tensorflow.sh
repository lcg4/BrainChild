# Change to "gpu" for GPU support.
TF_TYPE="cpu"

# Change to "darwin" for Mac OS.
OS="linux"

# Binary extraction folder.
TARGET_DIRECTORY="/usr/local"

# Download binary and install.
curl -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-${OS}-x86_64-1.4.1.tar.gz" | tar -C $TARGET_DIRECTORY -xz

# Configure the linker.
ldconfig
