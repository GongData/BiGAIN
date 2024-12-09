#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import logging
import random
import math
class ReadKddData:
    def __init__(self, data_path,max_samples=256):
        """Initialize data reader
        
        Args:
            data_path: path to CSV file
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # 基本参数初始化
        self.data_path = data_path
        self.batch_size = 128
        self.pointer = 0
        self.num_examples = 0
        self.maxLength = 96  # 时间序列最大长度
        self.isNormal = True  # 是否标准化数据
        
        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise IOError("Data file not found: {}".format(data_path))
        
        self.logger.info("Reading data from: {}".format(data_path))
        
        # 加载数据
        self._load_data()

        # 初始化 self.m 在限制数据量之前
        self.m = self.missing_mask.values  # 缺失值掩码
    
    

     
      
        
        # 初始化数据相关属性
        self.dic = list(range(self.num_features + 1))  # 特征字典
        
        # 生成时间序列相关数据
        self.times = np.arange(self.num_samples).reshape(-1, 1)  # 时间戳
        self.fileNames = ["sample_%d" % i for i in range(self.num_samples)]  # 文件名
        
        # 初始化标签和特征
        #self.y = np.zeros((self.num_samples, 1))  # 标签 要刪除!!!!!
        self.m = self.missing_mask.values  # 缺失值掩码
        
        # 初始化时间序列特征
        self.deltaPre = np.zeros((self.num_samples, self.num_features))  # 前向时间差
        self.deltaSub = np.zeros((self.num_samples, self.num_features))  # 后向时间差
        self.lastvalues = np.zeros((self.num_samples, self.num_features))  # 上一个值
        self.subvalues = np.zeros((self.num_samples, self.num_features))  # 下一个值
        
        # 序列长度初始化
        self.x_lengths = np.full(self.num_samples, self.maxLength)  # 每个样本的序列长度
        
        # 计算批次数
        self.num_batches = self.num_samples // self.batch_size
        
          # 如果指定了max_samples，限制数据量
        if max_samples is not None:
            self.logger.info("Limiting dataset to {} samples".format(max_samples))
            self.x = self.x[:max_samples]
            self.y = self.y[:max_samples]
            self.missing_mask = self.missing_mask.iloc[:max_samples]
            self.m = self.m[:max_samples]
            
            
            self.deltaPre = self.deltaPre[:max_samples]
            self.deltaSub = self.deltaSub[:max_samples]
            self.lastvalues = self.lastvalues[:max_samples]
            self.subvalues = self.subvalues[:max_samples]
            self.x_lengths = self.x_lengths[:max_samples]
            self.num_samples = min(self.num_samples, max_samples)
            self.num_examples = self.num_samples



        # 记录数据集信息
        self.logger.info("Dataset initialized:")
        self.logger.info(" - Number of samples: {}".format(self.num_samples))
        self.logger.info(" - Number of features: {}".format(self.num_features))
        self.logger.info(" - Number of batches: {}".format(self.num_batches))
        self.logger.info(" - Batch size: {}".format(self.batch_size))
        self.logger.info(" - Max sequence length: {}".format(self.maxLength))
        self.logger.info(" - Missing rate: {:.2f}%".format(np.mean(self.missing_mask.values)*100))
        
        # 计算每个特征的均值（忽略 NaN）
        self.mean = np.nanmean(self.x, axis=0)
        self.std = np.nanstd(self.x, axis=0)
        
        # 防止除零
        self.std[self.std == 0] = 1
        
        if self.isNormal:
            # 标准化数据
            self.x = (self.x - self.mean) / self.std
        '''    
        # 打印统计信息
        print("Data statistics after initialization:")
        print("Mean values:")
        for i, mean_val in enumerate(self.mean):
            print("Feature {}: {}".format(i, mean_val))
        print("\nStd values:")
        for i, std_val in enumerate(self.std):
            print("Feature {}: {}".format(i, std_val))
        '''
    def _load_data(self):
        """Load and preprocess data"""
        try:
            # Read CSV file
            df = pd.read_csv(self.data_path)
            
            # Remove time column if it exists
            if 'utc_time' in df.columns:
                df = df.drop('utc_time', axis=1)
            
            # 修改：检查是否存在污染物列，如果不存在则生成随机标签
            if not all(col in df.columns for col in ['PM2_5', 'PM10', 'NO2']):
                self.logger.warning("Pollution columns not found. Generating random labels.")
                # 生成随机标签（0和1），使用固定的随机种子以保持一致性
                np.random.seed(42)
                self.y = np.random.randint(0, 2, size=(len(df), 1))
            else:
                # 原有的标签生成逻辑
                is_polluted = ((df['PM2_5'] > 75) | 
                              (df['PM10'] > 150) | 
                              (df['NO2'] > 80))
                self.y = is_polluted.astype(int).values.reshape(-1, 1)
                
                # 去掉与污染标签直接相关的特征
                df = df.drop(columns=['PM2_5', 'PM10', 'NO2'], errors='ignore')

            # Handle categorical variables (like stationId)
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                # Handle NaN values before encoding
                df[col] = df[col].fillna('missing')
                df[col] = le.fit_transform(df[col])
            
            # Save missing value mask before filling NaN
            self.missing_mask = df.isna().astype(np.float32)
            
            # Handle numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                mean_val = df[col].mean()
                if pd.isna(mean_val):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(mean_val)
            
            # Convert all data to float32
            self.x = df.astype(np.float32).values

            # 在标准化之前检查数据
            self.logger.info("Data statistics before normalization:")
            self.logger.info("Min: {}, Max: {}, Mean: {}".format(
                np.min(self.x), np.max(self.x), np.mean(self.x)))
            
            # 检查是否有无穷大的值
            if np.any(np.isinf(self.x)):
                self.logger.warning("Found infinite values in data")
                self.x = np.clip(self.x, -1e10, 1e10)
            
            # 修改标准化方式，添加小的常数避免除零
            scaler = StandardScaler()
            self.x = scaler.fit_transform(self.x)
            
            # 再次检查标准化后的数据
            self.logger.info("Data statistics after normalization:")
            self.logger.info("Min: {}, Max: {}, Mean: {}".format(
                np.min(self.x), np.max(self.x), np.mean(self.x)))
            
            # 检查是否有 NaN
            if np.any(np.isnan(self.x)):
                self.logger.error("Found NaN values after normalization")
                # 将 NaN 替换为 0
                self.x = np.nan_to_num(self.x)
            
            # Save normalization parameters
            self.mean = scaler.mean_
            self.std = scaler.scale_
            
            # Update dimensions
            self.num_samples, self.num_features = self.x.shape
            self.num_examples = self.num_samples
            
            self.logger.info("Data loaded successfully:")
            self.logger.info(" - Number of samples: {}".format(self.num_samples))
            self.logger.info(" - Number of features: {}".format(self.num_features))
            self.logger.info(" - Missing rate: {:.2f}%".format(
                np.mean(self.missing_mask.values) * 100))
            
            # 验证标签
            unique_labels, counts = np.unique(self.y, return_counts=True)
            self.logger.info("Label distribution:")
            for label, count in zip(unique_labels, counts):
                self.logger.info("Label {}: {} samples ({:.2f}%)".format(
                    label, count, float(count)/len(self.y)*100))
            
            if len(unique_labels) != 2:
                self.logger.warning("Expected 2 unique labels, got {}: {}".format(
                    len(unique_labels), unique_labels))

        except Exception as e:
            self.logger.error("Error loading data: {}".format(str(e)))
            raise
        
    def set_batch_size(self, batch_size):
        """Set batch size"""
        self.batch_size = min(batch_size, self.num_samples)
        
    def nextBatch(self):
        """优化后的批次生成器方法"""
        print(" [*] Starting nextBatch generation...")
        
        i = 1
        while i * self.batch_size <= len(self.x):
            try:
                print(" [*] Processing batch {}".format(i))
                start_idx = (i-1) * self.batch_size
                end_idx = i * self.batch_size
                current_batch_size = end_idx - start_idx
                
                # 获取当前批次的数据切片
                batch_x = self.x[start_idx:end_idx]  # (batch_size, features)
                batch_m = self.m[start_idx:end_idx]  # (batch_size, features)
                
                # 获取其他数据并扩展维度
                batch_deltaPre = self.deltaPre[start_idx:end_idx]
                batch_lastvalues = self.lastvalues[start_idx:end_idx]
                batch_deltaSub = self.deltaSub[start_idx:end_idx]
                batch_subvalues = self.subvalues[start_idx:end_idx]
                y = self.y[start_idx:end_idx]
                x_lengths = self.x_lengths[start_idx:end_idx]
                #files = self.fileNames[start_idx:end_idx]

                # 擴展維度到 (batch_size, 96, features)
                batch_x = np.repeat(batch_x[:, np.newaxis, :], 96, axis=1)
                batch_m = np.repeat(batch_m[:, np.newaxis, :], 96, axis=1)
                batch_deltaPre = np.repeat(batch_deltaPre[:, np.newaxis, :], 96, axis=1)
                batch_lastvalues = np.repeat(batch_lastvalues[:, np.newaxis, :], 96, axis=1)
                batch_deltaSub = np.repeat(batch_deltaSub[:, np.newaxis, :], 96, axis=1)
                batch_subvalues = np.repeat(batch_subvalues[:, np.newaxis, :], 96, axis=1)
               
                # 关键修改：确保 data_mean 是一维数组 (n_features,)
                if self.isNormal:
                    mask = ~np.isnan(self.x)  # 创建非 NaN 的掩码
                    data_mean = np.zeros(self.num_features)

                    for i in range(self.num_features):
                        valid_data = self.x[:, i][mask[:, i]]
                        if len(valid_data) > 0:
                            data_mean[i] = np.mean(valid_data)
                        else:
                            data_mean[i] = 0    
                else:
                    '''
                    # 如果 self.mean 是多维的，取平均值转换为一维
                    if len(self.mean.shape) > 1:
                        data_mean = np.mean(self.mean, axis=(0, 1) if len(self.mean.shape) == 3 else 0)
                    else:
                        data_mean = self.mean
                    '''            
                    # 如果不需要标准化，使用原始数据的平均值
                    data_mean = np.nanmean(self.x, axis=0)  # 忽略 NaN 计算平均值                                
                # 确保没有 NaN 值
                data_mean = np.nan_to_num(data_mean)       

                # 擴展均值數據維度
                #data_mean = np.expand_dims(data_mean, axis=0)  # (1, features)
                #data_mean = np.expand_dims(data_mean, axis=0)  # (1, 1, features)
                #data_mean = np.repeat(data_mean, current_batch_size, axis=0)  # (batch_size, 1, features)
                #data_mean = np.repeat(data_mean, 96, axis=1)  # (batch_size, 96, features)
                
                # 處理時間相關特徵
                print(" [*] Processing time features...")
                imputed_deltapre = []
                imputed_deltasub = []
                imputed_m = []
                
                for j in range(current_batch_size):
                    one_imputed_deltapre = []
                    one_imputed_deltasub = []
                    one_G_m = []
                    
                    # 處理每個時間步
                    for h in range(x_lengths[j]):
                        if h == 0:
                            one_imputed_deltapre.append([0.0] * (len(self.dic)-1))
                            one_imputed_deltasub.append([1.0] * (len(self.dic)-1))
                            one_G_m.append([1.0] * (len(self.dic)-1))
                        elif h == x_lengths[j] - 1:
                            one_imputed_deltapre.append([1.0] * (len(self.dic)-1))
                            one_imputed_deltasub.append([0.0] * (len(self.dic)-1))
                            one_G_m.append([1.0] * (len(self.dic)-1))
                        else:
                            one_imputed_deltapre.append([1.0] * (len(self.dic)-1))
                            one_imputed_deltasub.append([1.0] * (len(self.dic)-1))
                            one_G_m.append([1.0] * (len(self.dic)-1))
                    
                    # 填充到最大長度
                    padding_length = self.maxLength - len(one_imputed_deltapre)
                    if padding_length > 0:
                        padding = [[0.0] * (len(self.dic)-1)] * padding_length
                        one_imputed_deltapre.extend(padding)
                        one_imputed_deltasub.extend(padding)
                        one_G_m.extend([[0.0] * (len(self.dic)-1)] * padding_length)
                    
                    imputed_deltapre.append(one_imputed_deltapre)
                    imputed_deltasub.append(one_imputed_deltasub)
                    imputed_m.append(one_G_m)
                
                # 轉換為numpy數組
                imputed_deltapre = np.array(imputed_deltapre)
                imputed_deltasub = np.array(imputed_deltasub)
                imputed_m = np.array(imputed_m)
                '''
                print(" [*] Final shapes:")
                print("     - X: {}".format(batch_x.shape))
                print("     - M: {}".format(batch_m.shape))
                print("     - Mean: {}".format(data_mean.shape))
                print("     - Delta Pre: {}".format(batch_deltaPre.shape))
                print("     - Delta Sub: {}".format(batch_deltaSub.shape))
                '''
                def check_array(arr, name):
                    if np.any(np.isnan(arr)):
                        print("Warning: Found NaN in {}".format(name))
                    if np.any(np.isinf(arr)):
                        print("Warning: Found Inf in {}".format(name))
                    return np.nan_to_num(arr)  # 将 NaN 和 Inf 替换为有限值
                                
                # 检查所有输出数据
                batch_x = check_array(batch_x, "batch_x")
                batch_m = check_array(batch_m, "batch_m")
                data_mean = check_array(data_mean, "data_mean")
                batch_deltaPre = check_array(batch_deltaPre, "batch_deltaPre")
                batch_lastvalues = check_array(batch_lastvalues, "batch_lastvalues")
                batch_deltaSub = check_array(batch_deltaSub, "batch_deltaSub")
                batch_subvalues = check_array(batch_subvalues, "batch_subvalues")
                                
                # 打印一些统计信息
                print("Batch statistics:")
                print("batch_x - min: {}, max: {}, mean: {}".format(
                    np.min(batch_x), np.max(batch_x), np.mean(batch_x)))
                print("data_mean - min: {}, max: {}, mean: {}".format(
                    np.min(data_mean), np.max(data_mean), np.mean(data_mean)))
                
                # 返回處理後的批次數據
                yield (batch_x, y, data_mean, batch_m, batch_deltaPre, 
                    x_lengths, batch_lastvalues, None, imputed_deltapre, 
                    imputed_m, batch_deltaSub, batch_subvalues, imputed_deltasub)
                
                i += 1
                
            except Exception as e:
                print(" [!] Error in batch {}: {}".format(i, str(e)))
                import traceback
                print(traceback.format_exc())
                raise

    def getAllData(self):
        """Get all data"""
        return self.x, self.missing_mask.values
    
    def inverse_transform(self, data):
        """Convert normalized data back to original scale"""
        return data * self.std + self.mean
        
    def get_statistics(self):
        """Get dataset statistics"""
        return {
            "num_samples": self.num_samples,
            "num_features": self.num_features,
            "missing_rate": np.mean(self.missing_mask.values)
        }
        
    def shuffle(self, batch_size, is_shuffle=True):
        """打乱数据集并设置批次大小
        
        Args:
            batch_size: int, 批次大小
            is_shuffle: bool, 是否打乱数据
        """
        if is_shuffle:
        # 將所有需要打亂的數據打包在一起
            c = list(zip(self.x, 
                        self.y,
                        self.missing_mask.values,
                        self.m,
                        self.deltaPre,
                        self.deltaSub,
                        self.lastvalues,
                        self.subvalues,
                        self.x_lengths))
            
            # 打亂數據
            random.shuffle(c)
        
            # 解包數據
            (self.x, 
            self.y,
            missing_mask_values,
            self.m,
            self.deltaPre,
            self.deltaSub,
            self.lastvalues,
            self.subvalues,
            self.x_lengths) = zip(*c)
            
            # 轉換回原來的數據類型
            self.x = np.array(self.x)
            self.y = np.array(self.y)
            self.missing_mask = pd.DataFrame(np.array(missing_mask_values))
            self.m = np.array(self.m)
            self.deltaPre = np.array(self.deltaPre)
            self.deltaSub = np.array(self.deltaSub)
            self.lastvalues = np.array(self.lastvalues)
            self.subvalues = np.array(self.subvalues)
            self.x_lengths = np.array(self.x_lengths)
    
        # 設置批次大小
        self.batch_size = min(batch_size, self.num_samples)
        self.num_batches = self.num_samples // self.batch_size

if __name__ == "__main__":
    dt = ReadKddData(
        #"../kdd/train_kdd3.csv"
        "../kdd/test_kdd3.csv"
    )
    print(dt.get_statistics())