from parameter import get_parameters
from patch_extraction import get_patch_list
from normalization import create_normalization_file
from train import train_model
from utils import setup_experiment, log_experiment


def main(config):
    experiment_name, current_time = setup_experiment(config.title, config)

    # normalization (creating t1_landmarks.npy file)
    create_normalization_file(
        use_controls=config.use_controls,
        use_nofcd=config.use_ae,
        mods=config.nb_of_modalities,
    )
    print('Normalization is finished')

    # patch extraction
    get_patch_list(
        use_controls=config.use_controls,
        use_fcd=config.use_ae,
        use_coronal=config.use_coronal,
        use_sagital=config.use_sagital,
        augment=config.augment,
        h=config.height,
        w=config.width,
        hard_labeling=config.hard_labeling,
        mods=config.nb_of_modalities,
        batch_size=config.batch_size
    )
    print('Patch extraction is finished')

    # cnn model
    top_k_scores = train_model(
        mods=config.nb_of_modalities,
        use_ae=config.use_ae,
        h=config.height,
        w=config.width,
        use_coronal=config.use_coronal,
        use_sagital=config.use_sagital,
        use_controls=config.use_controls,
        latent_dim=config.latent_size,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        weight_of_class=config.weight_of_class,
        n_epochs=config.nb_epochs,
        n_epochs_ae=config.nb_epochs_ae,
        p=config.dropout_rate,
        save_masks=config.save_masks,
        parallel=config.parallel,
        experiment_name=experiment_name,
        temporal_division=config.temporal_division,
        seed=config.seed
    )

    print(top_k_scores)
    print('LOO mean top-k score:', top_k_scores.mean())

    # logging
    log_experiment(config, current_time, (top_k_scores > 0).mean())


if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)
