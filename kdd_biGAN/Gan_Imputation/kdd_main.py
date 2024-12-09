#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
import numpy as np
import logging

# 获取项目根目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # Gan_Imputation目录
code_dir = os.path.dirname(current_dir)  # Code目录
sys.path.append(code_dir)

import WGAN_GRUI 
import tensorflow as tf
import argparse
from kdd.readData import ReadKddData

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='KDD数据集的GAN填补模型')
    # 解決梯度爆炸 : 改lr 改batch_size
    # 数据相关参数
    parser.add_argument('--data-path', type=str, default="../kdd/train_kdd3.csv")
    parser.add_argument('--test-data-path', type=str, default="../kdd/test_kdd3.csv",
                       help='测试数据集的路径')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dataset_name', type=str, default='kdd')
    parser.add_argument('--max-samples', type=int, default=256)  # 添加这一行 # 小數據測試(batch_size 整數倍)
    #parser.add_argument('--max-samples', type=int, default=None,  # 添加这一行
    #                   help='限制训练和测试数据集的最大样本数')
    # 模型相关参数
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--gen-length', type=int, default=96)
    parser.add_argument('--n-hidden-units', type=int, default=64)
    parser.add_argument('--n-classes', type=int, default=2)
    parser.add_argument('--z-dim', type=int, default=64)
    parser.add_argument('--n-inputs', type=int, default=10)  # 添加这一行，设置输入特征维度
    
    # 训练相关参数
    parser.add_argument('--impute-iter', type=int, default=400)
    parser.add_argument('--pretrain-epoch', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--g-loss-lambda', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--disc-iters', type=int, default=8)
    # 选择损失函数类型
    parser.add_argument('--loss-type', type=str, default='JS', choices=['JS', 'Original', 'Forward_KL', 'Reverse_KL', 'Pearson'],
                        help='选择损失函数类型: JS (default), Original, Forward KL, Reverse KL, Pearson')
    # 运行模式
    parser.add_argument('--run-type', type=str, default='train')
    
    # 路径相关
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint')
    parser.add_argument('--result-dir', type=str, default='results')
    parser.add_argument('--log-dir', type=str, default='logs')
    
    # 模型配置
    parser.add_argument('--isNormal', type=int, default=1)
    parser.add_argument('--isBatch-normal', type=int, default=1)
    parser.add_argument('--isSlicing', type=int, default=1)
    
    return parser.parse_args()

def main():
    # 设置日志
    logger = setup_logging()

    
    # 解析参数
    args = parse_arguments()
    
    # 处理布尔值参数
    args.isBatch_normal = (args.isBatch_normal == 1)
    args.isNormal = (args.isNormal == 1)
    args.isSlicing = (args.isSlicing == 1)

    # 定义超参数网格
    epochs = [30]
    g_loss_lambdas = [0.15]
    beta1s = [0.5]

    try:
        # 加载数据
        logger.info("Load Data : {}".format(args.data_path))
        train_dataset = ReadKddData(args.data_path)
        #logger.info("Initial train data label distribution:")
        #train_labels, train_counts = np.unique(train_dataset.y, return_counts=True)
        '''
        for label, count in zip(train_labels, train_counts):
            logger.info("Label {}: {} samples ({:.2f}%)".format(
                label, count, count/len(train_dataset.y)*100))
        '''
        
        logger.info("Load Test Data : {}".format(args.test_data_path))
        test_dataset = ReadKddData(args.test_data_path)
        #logger.info("Initial test data label distribution:")
        #test_labels, test_counts = np.unique(test_dataset.y, return_counts=True)
        '''
        for label, count in zip(test_labels, test_counts):
            logger.info("Label {}: {} samples ({:.2f}%)".format(
                label, count, count/len(test_dataset.y)*100))
        '''
        
        
        # 配置GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # 遍历所有超参数组合
        for beta1 in beta1s:
            for epoch in epochs:
                for g_loss_lambda in g_loss_lambdas:
                    #logger.info("\nTraining with parameters: epoch={}, beta1={}, g_loss_lambda={}".format(
                    #    epoch, beta1, g_loss_lambda)
                    
                    # 更新参数
                    args.epoch = epoch
                    args.beta1 = beta1
                    args.g_loss_lambda = g_loss_lambda
                    
                    # 创建新的 TensorFlow 会话
                    with tf.Session(config=config) as sess:
                        # 初始化WGAN模型
                        logger.info("initialize WGAN model")
                        gan = WGAN_GRUI.WGAN(sess, args=args, datasets=train_dataset)
                        try:
                            # 构建模型
                            logger.info("build model")
                            gan.build_model()
                 
                            # 训练模型
                            logger.info("start training")
                            gan.train()
                            logger.info("end training")
                            
                            # 对训练数据进行填补
                            logger.info("imputation train data")
                            imputed_data = gan.imputation(train_dataset, True)
                            
                            if imputed_data is not None:
                                imputed_data = train_dataset.inverse_transform(imputed_data)
                                #logger.info("train imputation result shape: {}".format(imputed_data.shape))
                                
                                if args.result_path:
                                    # 添加参数信息到文件名
                                    result_path = args.result_path.replace('.npy', 
                                        '_e{}_b{}_g{}.npy'.format(epoch, beta1, g_loss_lambda))
                                    np.save(result_path, imputed_data)
                                    logger.info("train result saved to: {}".format(result_path))
                        
                            # 对测试数据进行填补
                            logger.info("imputation test data") 
                            imputed_data = gan.imputation(test_dataset, False)
                            
                            if imputed_data is not None:
                                imputed_data = test_dataset.inverse_transform(imputed_data)
                                #logger.info("test imputation result shape: {}".format(imputed_data.shape))
                                
                                if args.result_path:
                                    # 添加参数信息到文件名
                                    test_result_path = args.result_path.replace('.npy', 
                                        '_test_e{}_b{}_g{}.npy'.format(epoch, beta1, g_loss_lambda))
                                    np.save(test_result_path, imputed_data)
                                    logger.info("test result saved to: {}".format(test_result_path))
                        finally:
                            # 确保会话关闭后再重置图
                            sess.close()
                    # 清理当前会话
                    tf.reset_default_graph()
          
    except Exception as e:
        logger.error("error: {}".format(str(e)), exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()