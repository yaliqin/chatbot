#! /bin/bash
#gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#for device in gpu_devices:
#    tf.config.experimental.set_memory_growth(device, True)

#CUDA_VISIBLE_DEVICES=0 python main.py

CUDA_VISIBLE_DEVICES=0 python dam_model.py
