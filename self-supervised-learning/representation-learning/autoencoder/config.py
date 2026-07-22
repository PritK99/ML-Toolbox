class Config:
    # Architecture related
    latent_dim = 1024

    # Dataset related
    batch_size = 64
    train_path = "ILSVRC/Data/CLS-LOC/train"
    test_path = "ILSVRC/Data/CLS-LOC/test"
    synset_mapping_file = "LOC_synset_mapping.txt"
    interpolation_dir = "interpolations"

    # Training hyperparameters
    num_epochs = 20  
    lr = 1e-4
    patience = 2