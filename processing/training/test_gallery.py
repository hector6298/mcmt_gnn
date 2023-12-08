import sys
import os
import os.path as osp
sys.path.insert(1, osp.abspath('.'))
import yaml
import torch
import torch_geometric.transforms as T
from datetime import date
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

from models.mcmt.gallery_rgnn import GalleryMOTMPNet
from modules.torch_dataset.object_graph_gallery_precomp import ObjectGraphGalleryPrecompDataset
from modules.torch_trainer.custom_loss import CECustom
from modules.plots import tsne2d_scatterplot, draw_pyg_network
from modules.inference.inference_module import InferenceModule
from modules.data_processor.annotations_processor import AnnotationsProcessor
from modules.tuning.epoch_functions import train_func, test_func_multiclass

with open("config/training_rgcnn.yml", "r") as config_file:
    config = yaml.safe_load(config_file)
    gnn_arch = config["gnn_arch"]

config = {
    "num_ids_per_graph": 64,
    "lr": 0.002,
    "epochs": 50 ,
    "warmup_duration":  3, # 0 is no warmup
    "optimizer_momentum":  0.9, # 0 is no momentum
    "lr_scheduler_step_size":  25, # 0 is scheduler off
    "ratio_neg_links_graph":  0.8 ,# 1 means original ratio
    "message_passing_steps": 1,
    "node_enc_fc_layers": 3, # Feat num decrease /2 every layer
    "edge_enc_fc_layers": 1, # Feat num increase x2 every layer
    "node_update_fc_layers": 1, # Channels icnrease x2 every layer
    "edge_update_fc_layers": 1,
    "edge_update_units": 20,
    "pruning": True,
    "spliting": True # Channels increase x2 every layer
}

sequence_path = 'C:/Users/mejia/Documents/tfm/mcmt_gnn/datasets/AIC20'
test_sequence = 'S02'
gt_filename = "gt_roi_filtered.txt"
sct_filename = "mtsc_tnt_ssd512_roi_filtered.txt"

def update_gnn_config(gnn_config, search_space_config):
    # GNN architecture configuration override by the hyperparam tuning
    gnn_config['num_enc_steps'] = search_space_config['message_passing_steps']

    # Node encoder
    gnn_config['encoder_feats_dict']['nodes']['node_fc_dims'] = [
        int(gnn_config['encoder_feats_dict']['nodes']['node_in_dim']//(2**i))\
            for i in range(1, search_space_config['node_enc_fc_layers']+1)
            ]
    node_out_dim = int(gnn_config['encoder_feats_dict']['nodes']['node_fc_dims'][-1]//2)
    gnn_config['encoder_feats_dict']['nodes']['node_out_dim'] = node_out_dim

    # Edge encoder
    gnn_config['encoder_feats_dict']['edges']['edge_fc_dims'] = [
        config["edge_update_units"]  for i in range(search_space_config['edge_enc_fc_layers'])
            ]
    edge_out_dim = gnn_config['encoder_feats_dict']['edges']['edge_fc_dims'][-1]
    gnn_config['encoder_feats_dict']['edges']['edge_out_dim'] = edge_out_dim

    # Node Update
    gnn_config['node_model_feats_dict']['fc_dims'] = [
        node_out_dim for i in range(search_space_config['node_update_fc_layers'])]
    
    # Edge update
    gnn_config['edge_model_feats_dict']['fc_dims'] = [
        edge_out_dim for i in range(search_space_config['edge_update_fc_layers'])]

    # Classifier update
    gnn_config['classifier_feats_dict']['edge_in_dim'] = gnn_config['edge_model_feats_dict']['fc_dims'][-1]

    gnn_config['gallery_combination'] = 'gru'
    gnn_config['combinator_feature_size'] = 2048
    gnn_config['combinator_output_size'] = 2048
    gnn_config['frames_per_gallery'] = 20
    gnn_config['combinator_num_gru_layers'] = 2

    return gnn_config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Instantiation of the dataset objects
print("Instantiation of the dataset objects")
train_dataset = ObjectGraphGalleryPrecompDataset(sequence_path_prefix=sequence_path,
                                sequence_names=["S01", "S03", "S04"],
                                annotations_filename='gt.txt',
                                num_ids_per_graph=config['num_ids_per_graph'],
                                return_dataframes=False,
                                negative_links_ratio=config['ratio_neg_links_graph'] \
                                    if config['ratio_neg_links_graph'] != 1 else None,
                                graph_transform=T.ToUndirected())

val_dataset = ObjectGraphGalleryPrecompDataset(sequence_path_prefix=sequence_path,
                                sequence_names=["S02"],
                                annotations_filename='gt.txt',
                                num_ids_per_graph=-1,
                                return_dataframes=False,
                                graph_transform=T.ToUndirected())

# Instantiation of the dataloaders
print("Instantiation of the dataloaders")
train_dataloader = DataLoader(train_dataset, 
                                batch_size=1, 
                                shuffle=False)
val_dataloader = DataLoader(val_dataset, 
                            batch_size=1, 
                            shuffle=False)

print("Creating the GNN model")
gnn_arch = update_gnn_config(gnn_arch, config)
gnn = GalleryMOTMPNet(gnn_arch, device).to(device)

optimizer = SGD(gnn.parameters(),
                lr=config["lr"],
                momentum=config["optimizer_momentum"])

print("Negative sampling: ", config["ratio_neg_links_graph"])
criterion = CrossEntropyLoss(reduction='mean') \
    if config["ratio_neg_links_graph"] != 1 else CECustom()


lr_scheduler = StepLR(optimizer, 
                        config["lr_scheduler_step_size"], 
                        0.9) if config["lr_scheduler_step_size"] else None

# Perform training
print("Perform training")
for epoch in tqdm(range(0, config['epochs'])):
    loss = train_func(gnn, optimizer, train_dataloader, criterion, lr_scheduler)
    print(loss)
    train_dataloader.dataset.on_epoch_end()
    metrics, graph, node_feats, edge_feats = test_func_multiclass(gnn, val_dataloader, criterion, 'multiclass')
    print(metrics)
