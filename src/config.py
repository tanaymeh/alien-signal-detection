class Config:
    N_SPLITS = 3
    model_name = 'vit_base_patch16_224'
    resize = (224, 224)
    TRAIN_BS = 32
    VALID_BS = 16
    num_workers = 8
    NB_EPOCHS = 10
    LABELS = 1
    FILE = "/input/train_labels.csv"
    FOLDER = "/input/train"