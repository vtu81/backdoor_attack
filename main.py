import argparse
import os
import pathlib
import torch
from data import PoisonedDataset, load_init_data, create_backdoor_data_loader
from deeplearning import backdoor_model_trainer
from models import BadNet, load_model
from utils import print_model_perform

parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='mnist', help='Which dataset to use (mnist or cifar10, default: mnist)')
parser.add_argument('--no_train', action='store_false', help='train model or directly load model (default true, if you add this param, then without training process)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optim', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--trigger_label', type=int, default=0, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--epoch', type=int, default=50, help='Number of epochs to train backdoor model, default: 50')
parser.add_argument('--batchsize', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--pp', action='store_true', help='Do you want to print performance of every label in every epoch (default false, if you add this param, then print)')
parser.add_argument('--datapath', default='./dataset/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--poisoned_portion', type=float, default=0.1, help='posioning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--mark_dir', default=None, help='trigger mark path (default None, the trigger would be a white square at the right bottom corner)')
parser.add_argument('--alpha', type=float, default=1.0, help='transparency of the trigger, only available when \'mark_dir\' is specified, default: 1.0')
parser.add_argument('--test_model_path', default=None, help='path to the model to be tested (default \'./checkpoints/badnet-<dataset>.pth\', only available when \'--no_train\' is activated)')

opt = parser.parse_args()

def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # create related path
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)

    print("\n# read dataset: %s " % opt.dataset)
    train_data, test_data = load_init_data(dataname=opt.dataset, device=device, download=opt.download, dataset_path=opt.datapath)

    print("\n# construct poisoned dataset")
    train_data_loader, test_data_ori_loader, test_data_tri_loader = create_backdoor_data_loader(opt.dataset, train_data, test_data, opt.trigger_label, opt.poisoned_portion, opt.batchsize, device, mark_dir=opt.mark_dir, alpha=opt.alpha)

    print("\n# begin training backdoor model")
    basic_model_path = "./checkpoints/badnet-%s.pth" % opt.dataset
    if opt.no_train:
        model = backdoor_model_trainer(
                dataname=opt.dataset,
                train_data_loader=train_data_loader, 
                test_data_ori_loader=test_data_ori_loader,
                test_data_tri_loader=test_data_tri_loader,
                trigger_label=opt.trigger_label,
                epoch=opt.epoch,
                batch_size=opt.batchsize,
                loss_mode=opt.loss,
                optimization=opt.optim,
                lr=opt.learning_rate,
                print_perform_every_epoch=opt.pp,
                basic_model_path= basic_model_path,
                device=device
                )
    else:
        if opt.test_model_path is None: opt.test_model_path = "./checkpoints/badnet-%s.pth" % opt.dataset
        model = load_model(opt.test_model_path
                                , model_type="badnet"
                                , input_channels=train_data_loader.dataset.channels
                                , output_num=train_data_loader.dataset.class_num
                                , device=device)


    print("\n# evaluation")
    print("## original test data performance:")
    print_model_perform(model, test_data_ori_loader)
    print("## triggered test data performance:")
    print_model_perform(model, test_data_tri_loader)

if __name__ == "__main__":
    main()
