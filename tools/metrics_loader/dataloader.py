import torch

from tools.metrics_loader.datasets import dataset_cortical_surfaces_segmentation


def loader_metrics_segmentation(data_path,
                                labels_path,
                                config,):

    ###############################################################
    #####################    TRAINING DATA    #####################
    ###############################################################

    train_dataset = dataset_cortical_surfaces_segmentation(config=config,
                                                            data_path=data_path,
                                                            labels_path=labels_path,
                                                            split='train',)

    #####################################
    ###############  UKB   ##############
    #####################################


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size = config['training']['bs'],
                                                shuffle = True, 
                                                num_workers=32)

    ###############################################################
    ####################    VALIDATION DATA    ####################
    ###############################################################


    val_dataset = dataset_cortical_surfaces_segmentation(data_path=data_path,
                                                config=config,
                                                labels_path=labels_path,
                                                split='val',)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=config['training']['bs_val'],
                                            shuffle=False,
                                            num_workers=32)
            

    ###############################################################
    #####################    TESTING DATA     #####################
    ###############################################################
            

    test_dataset = dataset_cortical_surfaces_segmentation(data_path=data_path,
                                            config=config,
                                            labels_path=labels_path,
                                            split='test',)


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=config['training']['bs_val'],
                                            shuffle=False, 
                                            num_workers=32)
    
    train_dataset.logging()


    print('')
    print('Training data: {}'.format(len(train_dataset)))
    print('Validation data: {}'.format(len(val_dataset)))
    print('Testing data: {}'.format(len(test_dataset)))

    return train_loader, val_loader, test_loader

