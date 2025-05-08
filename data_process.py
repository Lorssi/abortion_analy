import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# 设置日志文件
logger = logging.getLogger()

def calculate_l3_mean_and_variance(data):
    """
    计算每个三级单位的流产率均值和方差
    :param data: 输入数据
    :return: 处理后的数据
    """
    # 创建新的列来存储均值和方差
    data['abortion_1_7_l3_mean'] = np.nan
    data['abortion_1_7_l3_var'] = np.nan
        
    # 按日期和三级单位进行分组
    logger.info("开始计算三级单位的流产率均值和标准差")
    for (date, l3_name), group in tqdm(data.groupby(['date_code', 'l3_org_inv_nm'])):
        # 只有当组内有多个猪场时才需要计算
        if len(group) > 1:
            # 对于组内的每个猪场
            for idx, row in group.iterrows():
                current_farm = row['pigfarm_dk']
                    
                # 筛选该组内的其他猪场(同一天、同一三级单位、不同猪场)
                other_farms = group[group['pigfarm_dk'] != current_farm]
                    
                # 计算流产率的均值和方差
                mean_abortion = other_farms['abortion_1_7'].mean()
                var_abortion = other_farms['abortion_1_7'].std()
                    
                # 更新均值和方差字段
                data.at[idx, 'abortion_1_7_l3_mean'] = mean_abortion
                data.at[idx, 'abortion_1_7_l3_var'] = var_abortion
        
    print(f"{l2_name} 数据处理完成")
    return data

def pick_prrs_data(check_data):
    """
    筛选PRRS数据
    :param check_data: 输入数据
    :return: 筛选后的数据
    """
    # 筛选PRRS数据
    check_data = check_data[
        check_data['check_item_dk'].isin([
            # 野毒
            'bDoAAfRM6YiCrSt1',
            'bDoAArPPgj6CrSt1',
            'bDoAAfRM6IGCrSt1',
            'bDoAAfYsNUGCrSt1',
            'bDoAAfYsM8eCrSt1',
            'bDoAAfYr79SCrSt1',
            # 抗原
            'bDoAAJyZSTSCrSt1',
            'bDoAAfYgkW2CrSt1',
            'bDoAAfYq6LWCrSt1',
            'bDoAAfYq6kWCrSt1',
            'bDoAAfYsNKyCrSt1',
            'bDoAAwWyhPOCrSt1',
            # 抗体
            'bDoAAJyZSZiCrSt1'
        ])
    ]

    check_data = check_data[
        check_data['index_item_dk'].isin([
            # 野毒
            'bDoAAfYcdbLWD/D5',
            'bDoAAfYcdbTWD/D5',
            # 条带
            'bDoAAKqffmXWD/D5',
            'bDoAAKqewhjWD/D5',
            # 抗原
            'bDoAAfYq6kvWD/D5',
            # 抗体
            'bDoAAKqZiKzWD/D5',
            'bDoAAKqZiKzWD/D5',
            # s/p
            'bDoAAKqZiKzWD/D5',
            'bDoAAKqZiKzWD/D5'
        ])
    ]
    
    return check_data

def calculate_prrs_check_out_ratio(data, check_data):
    """
    计算PRRS检查结果的阳性率
    对于data中的每个猪场，计算其近15天内的PRRS检测阳性率
    阳性率 = sum(check_out_qty) / sum(check_qty)
    
    :param data: 输入数据
    :param check_data: 检查数据
    :return: 处理后的数据
    """
    # 创建新列存储PRRS检测阳性率
    data['prrs_check_out_ratio'] = np.nan
    
    # 确保check_data中的日期是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(check_data['date_code']):
        check_data['date_code'] = pd.to_datetime(check_data['date_code'])
    
    # 对data中的每一行数据进行处理
    logger.info("开始计算PRRS检测阳性率")
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        # 获取当前日期和猪场ID
        current_date = row['date_code']
        current_farm = row['pigfarm_dk']
        
        # 计算15天前的日期（包括当前日期，所以是14天前）
        date_15_days_ago = current_date - pd.Timedelta(days=14)
        
        # 筛选当前猪场在15天内的检测数据
        farm_recent_checks = check_data[
            (check_data['pigfarm_dk'] == current_farm) & 
            (check_data['date_code'] >= date_15_days_ago) & 
            (check_data['date_code'] <= current_date)
        ]
        
        # 如果找到检测数据，计算阳性率
        if not farm_recent_checks.empty:
            total_check_qty = farm_recent_checks['check_qty'].sum()
            total_check_out_qty = farm_recent_checks['check_out_qty'].sum()
            
            # 避免除零错误
            if total_check_qty > 0:
                check_out_ratio = total_check_out_qty / total_check_qty
                data.at[idx, 'prrs_check_out_ratio'] = check_out_ratio
    
    # 统计空值情况
    missing_ratio = data['prrs_check_out_ratio'].isna().mean() * 100
    print(f"PRRS检测阳性率缺失率: {missing_ratio:.2f}%")
    
    return data

if __name__ == "__main__":
    # 加载数据（如果已经加载则不需要重复）
    l2_name_list = ['猪业一部', '猪业二部', '猪业三部']
    
    # 确保输出目录存在
    os.makedirs('processed_data', exist_ok=True)
    
    for l2_name in l2_name_list:
        print(f"处理 {l2_name} 的数据...")
        data = pd.read_csv(f'raw_data/ads_pig_org_total_to_ml_training_day_abortion_{l2_name}.csv', encoding='utf-8')
        check_data = pd.read_csv('raw_data\TMP_PIG_ORG_DISEASE_CHECK_RESULT_DAY_prrs.csv', encoding='utf-8')

        data = data.rename(columns={'stats_dt': 'date_code'})
        data['date_code'] = pd.to_datetime(data['date_code'])

        check_data = check_data.rename(columns={'receive_dt': 'date_code'})
        check_data = check_data.rename(columns={'org_inv_dk': 'pigfarm_dk'})
        check_data['date_code'] = pd.to_datetime(check_data['date_code'])
        check_data = pick_prrs_data(check_data)

        data = calculate_l3_mean_and_variance(data)
        data = calculate_prrs_check_out_ratio(data, check_data)

        # 保存处理后的数据
        data.to_csv(f'processed_data/ads_pig_org_total_to_ml_training_day_abortion_{l2_name}_processed.csv', index=False, encoding='utf-8')
        
    
    print("所有数据处理完成")

