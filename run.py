
import argparse
import train
import train_backup
import test
import evaluate

if __name__ == '__main__':
    
    #? IDK what to use these for. Were mentioned in run_experiment.sh
    GPU_IDX=0
    CUDA_DEVICE_ORDER="PCI_BUS_ID"
    CUDA_VISIBLE_DEVICES=GPU_IDX
    
    
    #? setup arg parser to pass in config info
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='./log/', help='path to model save dir')
    parser.add_argument('--loglevel', type=str, default='info', help='set level of logger')
    parser.add_argument('--identifier', type=str, default='debug', help='unique run identifier')
    parser.add_argument('--config', type=str, default='./configs/dfaust/config_dfaust.yaml', help='path to yaml config file')
    parser.add_argument('--model_ckpt', type=str, default='000000.pt', help='checkpoint to load')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix random seed')
    args = parser.parse_args([])

    #? Manually set arg parser values here:
    #? Comment out if you want to use terminal flags above instead (I just got tired of manually entering them in)
    args.logdir = './log/'
    args.loglevel = 'debug'
    args.identifier = 'debug_msr_action_backup'
    #args.identifier = 'set_transformer_dbug'
    args.config = r'configs/msr-action3d/config_msr_action3d.yaml'
    args.model_ckpt = '000001.pt'
    args.fix_random_seed = True
    print('args', args)
    
    print('--------------- starting training')
    train.main(args) #--identifier $IDENTIFIER --config $CONFIG --logdir $LOGDIR --fix_random_seed
    #train_backup.main(args)
   
    #print('--------------- starting testing')
    # TODO: Needs fixing
    #test.main(args) #--identifier $IDENTIFIER --model_ckpt '000001.pt' --logdir $LOGDIR --fix_random_seed
    #
    #print('--------------- starting eval')
    #evaluate.main(args) #--identifier $IDENTIFIER --logdir $LOGDIR