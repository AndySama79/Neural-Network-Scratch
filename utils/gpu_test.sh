#!	/usr/bin/bash

conda init

# activate conda environment containing tensorflow
conda activate tiramisu
python -c "import tensorflow as tf; tf.config.list_physical_devices()"

# deactivate the env
conda deactivate

# activate conda environment containing pytorch
conda activate pikachu
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"

# deactivate the env
conda deactivate

