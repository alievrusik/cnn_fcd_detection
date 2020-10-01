python main.py \
--height 24 --width 40 --temporal_division --use_sagital --use_coronal \
--use_ae --batch_size 128  \
--augment --lr 3e-4 --latent_size 256 --nb_epochs 2 --nb_epochs_ae 2 --weight_decay 4e-4 \
--weight_of_class 1 --dropout_rate 0.4 --save_masks --title 'patchmodel'