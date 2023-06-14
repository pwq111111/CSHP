import os
import optuna
import torch.utils.data as Data
import torch
import sys
from prettytable import PrettyTable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.get_data import *
from data.data_process import *

import ppt_model
import config
import time

def main():

    logger = get_logger(logger_name=args.name, log_dir='./result/')
    logger.info('\n' +'epoch:' + str(args.epoch) +'  seed:' + str(args.seed) + '  attention_num:' +str(args.attention_num) +
                '  prompt_position:'+ str(args.prompt_position) + '  prompt_length:' + str(args.prompt_length) + '  hard_prompt_front:' +
                args.hard_prompt_front + '  hard_prompt_back:' +args.hard_prompt_back)
    data_start_time = time.time()
    class_name, class_name1,class_name2,train_loader, test_loader, test_loader1, random_num, random_num1 = args.data['data'](args)

    prompt_coop = 'a '*args.prompt_length
    hard_prompt_front = prompt_coop + args.hard_prompt_front
    train_start_time=time.time()
    # torch.manual_seed(args.seed_cpu)
    # torch.cuda.manual_seed(args.seed_gpu)
    model = ppt_model.model_prompt(args.batch_size,args.prompt_length, args.embedding_length, args.prompt_position, args.prompt_cls_length,
                                    args.prompt_cls_position, args.token_length, prompt_coop,args.attention_num).cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    model.train()
    ppt_model.train(model, train_loader,hard_prompt_front,
                    args.hard_prompt_back, args.epoch, class_name, optimizer,logger)
    test_start_time = time.time()
    model.eval()
    right_number0 = ppt_model.test(model, test_loader, class_name1, args.class_min, hard_prompt_front, args.hard_prompt_back,
                                   args.class_num)
    right_number1 = ppt_model.test(model, test_loader1, class_name2, args.class_min, hard_prompt_front, args.hard_prompt_back,
                                   args.class_num1)
    # print_table.add_row([time.time() - t1, args.epoch, args.seed, args.attention_num,args.learning_rate,args.prompt_position, args.prompt_length, args.hard_prompt_front,
    #                      args.hard_prompt_back, 0, right_number0, right_number1,(right_number0*random_num + right_number1*random_num1) /(random_num1+random_num)])
    logger.info('data cost=' + '{:.2f}'.format(train_start_time-data_start_time) +'  train cost=' +'{:.2f}'.format(test_start_time-train_start_time)
                +'  test cost=' +'{:.2f}'.format(time.time()-test_start_time) + '  acc1='+'%s%%'%('{:.2f}'.format(right_number0.item()*100)) + '   acc2=' +'%s%%'%('{:.2f}'.format(100*right_number1.item()))
                + '  average= ' + '%s%%'%('{:.2f}'.format(100*(right_number0.item()*random_num + right_number1.item()*random_num1) /(random_num1+random_num))))


if __name__ == "__main__":

    args = config.get_args_imagenet()

    main()

