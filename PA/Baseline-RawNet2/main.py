import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from model import RawNet
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed

__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"
__credits__ = ["Jose Patino", "Massimiliano Todisco", "Jee-weon Jung"]


def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y in dev_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
    model.eval()
    
    for batch_x,utt_id in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1]
                       ).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))

def train_epoch(train_loader, model, lr,optim, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in train_loader:
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct/num_total)*100))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
       
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/your/path/to/data/ASVspoof_database/', help='Change this to user\'s full directory address of PA database (ASVspoof2019- for training & validation, ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 PA train, PA dev and ASVspoof2021 PA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- PA
    %      |- ASVspoof2021_PA_eval/flac
    %      |- ASVspoof2019_PA_train/flac
    %      |- ASVspoof2019_PA_dev/flac
    '''

    parser.add_argument('--protocols_path', type=str, default='/your/path/to/protocols/ASVspoof_database/', help='Change with path to user\'s PA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_PA_cm_protocols
    %      |- ASVspoof2021.PA.cm.eval.trl.txt
    %      |- ASVspoof2019.PA.cm.dev.trl.txt 
    %      |- ASVspoof2019.PA.cm.train.trn.txt 
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='PA',choices=['LA', 'PA','DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 
    

    dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'

    with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.load(f_yaml)

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    track = args.track

    assert track in ['LA', 'PA','DF'], 'Invalid track given'    

    # Database
    prefix      = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    

    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
   

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    # model 
    model = RawNet(parser1['model'], device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =(model).to(device)
    
    # set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    
    # evaluation 
    if args.eval:
        # edit+++++++
        # original:
        # file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.eval.trl.txt'.format(prefix,prefix_2021)),is_train=False,is_eval=True)
        # print('no. of eval trials',len(file_eval))
        file_eval_all = genSpoof_list(dir_meta=os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.eval.trl.txt'.format(prefix,prefix_2021)), is_train=False, is_eval=True)
        eval_base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(args.track))
        eval_flac_dir = os.path.join(eval_base_dir, 'flac')
        file_eval = []
        for utt_id in file_eval_all:
            if os.path.isfile(os.path.join(eval_flac_dir, utt_id + '.flac')):
                file_eval.append(utt_id)
                if len(file_eval) >= 1000:
                    break
        print('no. of eval trials (requested)', 1000)
        print('no. of eval trials (found on disk)', len(file_eval))
        if len(file_eval) == 0:
            raise FileNotFoundError('No evaluation FLAC files found under {}'.format(eval_flac_dir))
        # edit+++++++
        eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = eval_base_dir)
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)

    # define train dataloader

    d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.train.trn.txt'.format(prefix,prefix_2019)),is_train=True,is_eval=False)
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_ASVspoof2019_train(list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_{}_train/'.format(args.track)))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn
    

    # define validation dataloader

    d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.dev.trl.txt'.format(prefix,prefix_2019)),is_train=False,is_eval=False)
    print('no. of validation trials',len(file_dev))

    dev_set = Dataset_ASVspoof2019_train(list_IDs = file_dev,
		labels = d_label_dev,
		base_dir = os.path.join(args.database_path+'ASVspoof2019_{}_dev/'.format(args.track)))
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)
    del dev_set,d_label_dev

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    best_acc = 99
    for epoch in range(num_epochs):
        running_loss, train_accuracy = train_epoch(train_loader,model, args.lr,optimizer, device)
        valid_accuracy = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {:.2f} - {:.2f}'.format(epoch,
                                                   running_loss, train_accuracy, valid_accuracy))
        
        if valid_accuracy > best_acc:
            print('best model find at epoch', epoch)
        best_acc = max(valid_accuracy, best_acc)
        torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))


# protocal path example:
# /tmp2/xwang1/datasets/ASVspoof2019/PA/ASVspoof_PA_cm_protocols/ASVspoof2021.PA.cm.eval.trl.txt
# this 2021 is only contains the 00.tar, not all the eve data. the full key is stored in the same folder.
# only evel 1000 trials are used for evaluation to speed up the process. you can change it to all the eval trials if you want.
# command to use pretrained model for evaluation on ASVspoof2021 PA eval set (change the paths and model checkpoint as needed):

# bsub -q gpu -R "select[(ostype=='any')]" \
# -gpu num=1:gmem=10000:gmodel=NVIDIAL40S:mode=shared:j_exclusive=yes \
# -o eval_output.log -e eval_error.log -J PA_eval \
# python main.py \
#   --database_path /tmp2/xwang1/datasets/ASVspoof2019/PA/ \
#   --protocols_path /tmp2/xwang1/datasets/ASVspoof2019/PA/ \
#   --track PA \
#   --eval \
#   --model_path /tmp2/xwang1/models/pre_trained_DF_RawNet2.pth \
#   --eval_output results/eval_scores1.txt