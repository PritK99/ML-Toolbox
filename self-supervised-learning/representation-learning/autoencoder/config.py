class Config:
    # Architecture related
    latent_dim = 512

    # Dataset related
    cow_json_path = "../../../data/full_simplified_cow.ndjson"
    bulldozer_json_path = "../../../data/full_simplified_bulldozer.ndjson"
    val_ratio = 0.1

    # Training hyperparameters
    num_epochs = 15    
    batch_size = 32
    lr = 1e-4
    patience = 2