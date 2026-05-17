import torch
from torch.utils.data import DataLoader
from modules.DeepAttn_MIL.deep_attn_mil import DeepAttnMIL_Surv
from utils.process_utils import get_process_pipeline
from utils.wsi_utils import WSI_Dataset
from utils.general_utils import set_global_seed, init_epoch_info_log, add_epoch_info_log, early_stop
from utils.model_utils import get_optimizer, get_scheduler, get_criterion, save_last_model, save_log, model_select
from utils.loop_utils import train_loop_deep_attn_mil, val_loop
from tqdm import tqdm


def process_DeepAttn_MIL(args):
    """
    Training process for DeepAttnMIL on WSI classification
    
    Args:
        args: Configuration from YAML file
    """
    
    # ============ Load Datasets ============
    train_dataset = WSI_Dataset(args.Dataset.dataset_csv_path, 'train', args)
    val_dataset = WSI_Dataset(args.Dataset.dataset_csv_path, 'val', args)
    test_dataset = WSI_Dataset(args.Dataset.dataset_csv_path, 'test', args)

    process_pipeline = get_process_pipeline(val_dataset, test_dataset)
    args.General.process_pipeline = process_pipeline
    
    # ============ Setup Data Loaders ============
    generator = torch.Generator()
    generator.manual_seed(args.General.seed)
    set_global_seed(args.General.seed)
    num_workers = args.General.num_workers
    use_balanced_sampler = args.Dataset.balanced_sampler.use
    
    if use_balanced_sampler:
        sampler = train_dataset.get_balanced_sampler(replacement=args.Dataset.balanced_sampler.replacement)
        train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=num_workers, 
                                     generator=generator, sampler=sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
                                     num_workers=num_workers, generator=generator)
    
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    
    print('DataLoader Ready!')
    
    # ============ Initialize Model ============
    device = torch.device(f'cuda:{args.General.device}')
    num_classes = args.General.num_classes
    
    # Extract model parameters
    in_dim = args.Model.in_dim
    embedding_dim = args.Model.embedding_dim
    attention_dim = args.Model.attention_dim
    fc_dim = args.Model.fc_dim
    dropout = args.Model.dropout
    cluster_num = args.Model.cluster_num
    act = args.Model.act
    l1_lambda = args.Model.l1_lambda
    
    # Create model
    mil_model = DeepAttnMIL_Surv(
        in_dim=in_dim,
        embedding_dim=embedding_dim,
        attention_dim=attention_dim,
        fc_dim=fc_dim,
        num_classes=num_classes,
        dropout=dropout,
        cluster_num=cluster_num,
        act=act
    )
    mil_model.to(device)
    
    print('Model Ready!')
    print(f'DeepAttnMIL Configuration:')
    print(f'  - Input Dim: {in_dim}')
    print(f'  - Embedding Dim: {embedding_dim}')
    print(f'  - Attention Dim: {attention_dim}')
    print(f'  - FC Dim: {fc_dim}')
    print(f'  - Num Classes: {num_classes}')
    print(f'  - Cluster Num: {cluster_num}')
    print(f'  - Dropout: {dropout}')
    print(f'  - Activation: {act}')
    print(f'  - L1 Lambda: {l1_lambda}')
    
    # ============ Setup Training Components ============
    optimizer, base_lr = get_optimizer(args, mil_model)
    
    # Handle scheduler - check if it returns tuple or single value
    scheduler_result = get_scheduler(args, optimizer, base_lr)
    if isinstance(scheduler_result, tuple):
        scheduler, warmup_scheduler = scheduler_result
    else:
        scheduler = scheduler_result
        warmup_scheduler = None
    
    criterion = get_criterion(args.Model.criterion)
    warmup_epoch = args.Model.scheduler.warmup
    
    # ============ Training Loop ============
    epoch_info_log = init_epoch_info_log()
    best_model_metric = args.General.best_model_metric
    REVERSE = False
    best_val_metric = 0
    
    if best_model_metric in ['val_loss', 'loss']:
        REVERSE = True
        best_val_metric = 9999
    
    best_epoch = 1
    print('Start Process!')
    print('Using Process Pipeline:', process_pipeline)
    
    for epoch in tqdm(range(args.General.num_epochs), colour='GREEN'):
        # Select scheduler
        if warmup_scheduler is not None and epoch + 1 <= warmup_epoch:
            now_scheduler = warmup_scheduler
        else:
            now_scheduler = scheduler
        
        # Training
        train_loss, cost_time = train_loop_deep_attn_mil(device, mil_model, train_dataloader, criterion, optimizer, now_scheduler, l1_lambda=l1_lambda)
        
        # Validation/Testing
        if process_pipeline == 'Train_Val_Test':
            val_loss, val_metrics = val_loop(device, num_classes, mil_model, val_dataloader, criterion)
            test_loss, test_metrics = val_loop(device, num_classes, mil_model, test_dataloader, criterion)
        elif process_pipeline == 'Train_Val':
            val_loss, val_metrics = val_loop(device, num_classes, mil_model, val_dataloader, criterion)
            test_loss, test_metrics = None, None
        elif process_pipeline == 'Train_Test':
            val_loss, val_metrics, test_loss, test_metrics = None, None, None, None
            if epoch + 1 == args.General.num_epochs:
                test_loss, test_metrics = val_loop(device, num_classes, mil_model, test_dataloader, criterion)
        
        # Print epoch info
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        print('----------------INFO----------------\n')
        print(f'{FAIL}EPOCH:{ENDC}{epoch+1},  Train_Loss:{train_loss},  Val_Loss:{val_loss},  Test_Loss:{test_loss},  Cost_Time:{cost_time}\n')
        print(f'{FAIL}Val_Metrics:  {ENDC}{val_metrics}\n')
        print(f'{FAIL}Test_Metrics:  {ENDC}{test_metrics}\n')
        add_epoch_info_log(epoch_info_log, epoch, train_loss, val_loss, test_loss, val_metrics, test_metrics)
        
        # Model selection
        best_val_metric, best_epoch = model_select(
            REVERSE, args, mil_model.state_dict(), val_metrics,
            best_model_metric, best_val_metric, epoch, best_epoch
        )
        
        # Early stopping
        if early_stop(args, epoch_info_log, process_pipeline, epoch, mil_model.state_dict(), best_epoch):
            break
        
        # Save final model
        if epoch + 1 == args.General.num_epochs:
            save_last_model(args, mil_model.state_dict(), epoch + 1)
            save_log(args, epoch_info_log, best_epoch, process_pipeline)