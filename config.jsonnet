// jsonnet allows local variables like this
local embedding_dim = 6;
local hidden_dim = 6;
local num_epochs = 10;
local patience = 10;
local batch_size = 2;
local learning_rate = 0.005;
local weight_decay = 0.0001;

{
    "train_data_path": 'train_data.txt',
    "validation_data_path": 'val_data.txt',
    "dataset_reader": {
        "type": "nameLang-dataset"
    },
    "model": {
        "type": "lstm-labeller",
        "char_embeddings": {
            "token_embedders": {
                "characters": {
                    "type": "embedding",
                    "embedding_dim": embedding_dim
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": hidden_dim
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": batch_size
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "adam",
            "lr": learning_rate,
            "weight_decay": weight_decay
        },
        "patience": patience,
        "cuda_device": 0
    }
}