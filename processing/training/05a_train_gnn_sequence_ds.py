import sys
import os
import os.path as osp
sys.path.insert(1, osp.abspath('.'))
import yaml
import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
import torchvision.transforms.v2 as transforms
from torch_geometric.loader import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from modules.data_processor.annotations_processor import AnnotationsProcessor
from models.reid.resnet import resnet101_ibn_a
from modules.torch_trainer.trainer import TrainingEngineRGNNBinary, TrainingEngineRGNNMulticlass
from modules.torch_trainer.custom_loss import focal_loss, CECustom
from modules.torch_dataset.object_graph_dataset import ObjectGraphDataset
from models.mcmt.rgnn import MOTMPNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set a global random seed for CPU
torch.manual_seed(11)
np.random.seed(11)

# Set a global random seed for CUDA (GPU) if available
if torch.cuda.is_available():
    torch.cuda.manual_seed(11)

    # Additional CUDA configurations for reproducibility (optional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Initialization of configuration objects
with open("config/training_rgcnn.yml", "r") as config_file:
    config = yaml.safe_load(config_file)
    gnn_arch = config["gnn_arch"]
    training_config = config["training_config"]
    optimizer_config = config["training_config"]["optimizer"]
    lr_scheduler_config = config["training_config"]["lr_scheduler"]
    loss_config = config["training_config"]["loss"]
    dataset_config = config["training_config"]['dataset']
    reid_config = training_config['reid']
print("Success Opening configuration file.")

if __name__ == "__main__":

    # Loading REID model to generate embeddings for both train and test
    reid_model = resnet101_ibn_a(model_path=reid_config['reid_model_path'], device=device)

    # Consolidating annotations from all cameras of the training sequences into a single dataframe
    train_df = AnnotationsProcessor(
            sequence_path=dataset_config['sequence_path_prefix'],
            annotations_filename=dataset_config['annotations_filename']
        ).consolidate_annotations(dataset_config['sequences_train'])

    # Augmentation engine for the training dataset
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.02, 0.15)),
        transforms.RandomRotation(degrees=0.15),
        transforms.RandomPerspective(distortion_scale=0.25)
    ])

    if training_config['debug_run']:
        ids = np.random.choice(train_df.id.unique(), 5)
        train_df = train_df[train_df.id.isin(ids)]

    # Instantiating the Graph dataset using the detections of data_df 
    train_dataset = ObjectGraphDataset(train_df, 
                                       sequence_path_prefix=dataset_config['sequence_path_prefix'], 
                                       reid_model=reid_model,
                                       num_ids_per_graph=dataset_config['num_ids_per_graph'], 
                                       embeddings_per_it=dataset_config['embeddings_per_iteration'], 
                                       resized_img_shape=dataset_config['resized_img_shape'], 
                                       augmentation=augmentation,
                                       return_dataframes=False,
                                       negative_sampling=dataset_config['negative_sampling'],
                                       graph_transform=T.ToUndirected())
    
    # Consolidating annotations from all cameras in S02 into a single dataframe
    test_df = AnnotationsProcessor(
            sequence_path=dataset_config['sequence_path_prefix'],
            annotations_filename=dataset_config['annotations_filename']
        ).consolidate_annotations(dataset_config['sequences_val'])

    if training_config['debug_run']:
        ids = np.random.choice(test_df.id.unique(), 5)
        test_df = test_df[test_df.id.isin(ids)]

    # Instantiating the Graph dataset using the detections of data_df 
    # Test dataset has NO augmentation and NO sampling. All objects loaded at once
    val_dataset = ObjectGraphDataset(test_df, 
                                      sequence_path_prefix=dataset_config['sequence_path_prefix'], 
                                      reid_model=reid_model,
                                      num_ids_per_graph=-1, # -1 means load all ids
                                      embeddings_per_it=dataset_config['embeddings_per_iteration'], 
                                      resized_img_shape=dataset_config['resized_img_shape'], 
                                      augmentation=None, # No augmentation for validation of course
                                      return_dataframes=False,
                                      transform=T.ToUndirected()) 
    print("Success Defining torch datasets.")

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=dataset_config['train_batch_size'], 
                                  shuffle=False)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=dataset_config['test_batch_size'], 
                                shuffle=False)
    print("Success creating torch dataloaders.")

    # 2 - Initializing Graph Neural Network
    gnn = MOTMPNet(gnn_arch).to(device)
    print("GNN defined.")

    # 3 - Initialing optimizer
    optimizer = SGD(gnn.parameters(),
                    lr=optimizer_config["lr"],
                    momentum=optimizer_config["momentum"],
                    dampening=optimizer_config["dampening"])
    print("Success defining optimizer.")


    # 4a - Binary Cross Entropy loss
    if loss_config['loss_type'] == 'bce':
        criterion = BCEWithLogitsLoss(reduction='mean', 
                                    pos_weight=torch.tensor(loss_config['loss_weight']) 
                                            if 'loss_weight' in loss_config else None)
        TrainingEngine = TrainingEngineRGNNBinary

    # 4b - Cross Entropy loss with optional fixed weights
    elif loss_config["loss_type"] == 'ce':
        weights = (torch.tensor(loss_config['loss_weight']).to(device)
                    if 'loss_weight' in loss_config else None)
        criterion = CrossEntropyLoss(weight=weights, reduction='mean')
        TrainingEngine = TrainingEngineRGNNMulticlass

    # 4c - Focal Loss
    elif loss_config["loss_type"] == 'focal':
        criterion = focal_loss(loss_config['alpha'], loss_config['gamma'])
        TrainingEngine = TrainingEngineRGNNMulticlass

    # 4d - Default is custom CrossEntropy Loss
    else:
        criterion = CECustom()
        TrainingEngine = TrainingEngineRGNNMulticlass
    print(f"Success creating loss function for {loss_config['loss_type']}")


    # 5 - Learning rate decay scheduler
    if lr_scheduler_config['toggle']:
        if lr_scheduler_config["scheduler_type"] == 'step':
            lr_scheduler = StepLR(optimizer, 
                                lr_scheduler_config["step_size"], 
                                lr_scheduler_config["gamma"])
            
        elif lr_scheduler_config["scheduler_type"] == 'cosine':
            lr_scheduler = CosineAnnealingLR(optimizer,
                                            T_max=lr_scheduler_config["tmax"])
        else:
            raise Exception(f'Learning rate scheduler {lr_scheduler_config["scheduler_type"]} is not supported.')
        print(f"Success creating learning rate scheduler: {lr_scheduler_config['scheduler_type']}")
    else:
        lr_scheduler = None
        
    # 6 - Definition  of training engine using our custom object
    training_engine = TrainingEngine(train_dataloader, val_dataloader, gnn, training_config["checkpoint_path_prefix"], device)
    training_engine.setup_engine(
        optimizer=optimizer,
        lr=optimizer_config["lr"],
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        warmup_duration=lr_scheduler_config["warmup_duration"],
        show_training_progress=True,
        metrics=["Accuracy", "Precision", "Recall"],
        resume_training_flag=training_config["load_previous_train_state"],
        use_tensorboard=True,
        log_training=True
    )

    # Check if there are previous checkpoints. Resume training if there are
    if os.path.exists(training_config["checkpoint_path_prefix"]):
        if len(os.listdir(osp.join(training_config["checkpoint_path_prefix"], "checkpoints"))) and training_config["load_previous_train_state"]:
            training_engine.load_train_state()

    # Run or resume training
    training_engine.run_training(max_epochs=1 if training_config['debug_run'] else training_config["epochs"])