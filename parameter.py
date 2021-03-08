import argparse


def get_parameters():
    parser = argparse.ArgumentParser()

    # experiment setting
    parser.add_argument('--use_controls', action='store_true', default=False,
                        help='Use control subjects (LA5 and HCP dataset) for model training')
    parser.add_argument('--use_ae', action='store_true', default=True,
                        help='Pretrain classifier with autoencoder')
    parser.add_argument('--nb_of_modalities', default=3, type=int,
                        help='How many modalities to use for training (T1, T2, FLAIR)')
    parser.add_argument('--temporal_division', action='store_true', default=False,
                        help='Train temporal and non-temporal models separately')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for model training')

    # patch extraction
    parser.add_argument('--height', default=40, type=int, help='Height of patches (should be divisible by 8)')
    parser.add_argument('--width', default=64, type=int, help='Width of patches (should be divisible by 8)')
    parser.add_argument('--use_coronal', action='store_true', default=False, help='Use coronal patches')
    parser.add_argument('--use_sagital', action='store_true', default=False, help='Use sagital patches')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Augment with oversampling patches near FCD region')
    parser.add_argument('--hard_labeling', action='store_true', default=False,
                        help='Use hard labels (0/1) instead of probabilities')

    # model setting
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--latent_size', default=256, type=int, help='Size of latent code of autoencoder')
    parser.add_argument('--nb_epochs', default=4, type=int)
    parser.add_argument('--nb_epochs_ae', default=4, type=int)
    parser.add_argument('--weight_decay', default=4e-4, type=float)
    parser.add_argument('--weight_of_class', default=1, type=float,
                        help='Weight of 1st class for cross entropy loss')
    parser.add_argument('--dropout_rate', default=0.4, type=float)
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='Data parallelism training')

    # logging
    parser.add_argument('--save_masks', action='store_true', default=True,
                        help='Save predicted masks')
    parser.add_argument('--title', default='patch_model', type=str,
                        help='Title of experiment')
    parser.add_argument('--message', default='Test run', type=str,
                        help='Notes on current experiment')

    return parser.parse_args()
