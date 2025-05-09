import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def draw_pigfarm_all_years(mydata, base_output_dir, asf_data):
    
    # 按猪场分组处理数据
    for org_inv_nm, org_group in tqdm(mydata.groupby('org_inv_nm')):
        
        # 检查该猪场的流产率是否全为空
        if org_group['abortion_1_7'].isna().all():
            continue  # 跳过流产率全为空的猪场
            
        # 根据流产率最大值设置图表参数
        if org_group['abortion_1_7'].max() >= 0.05:
            figsize_height = 16
            y_max = 0.1
            y_interval = 0.01
            prefix = '3'
        elif org_group['abortion_1_7'].max() >= 0.005:
            figsize_height = 16
            y_max = 0.05
            y_interval = 0.005
            prefix = '2'
        elif org_group['abortion_1_7'].max() >= 0.0025:
            figsize_height = 16
            y_max = 0.05
            y_interval = 0.005
            prefix = '1'
        else:
            figsize_height = 16
            y_max = 0.05
            y_interval = 0.005
            prefix = '0'

        # 处理日期格式
        org_group['date_code'] = pd.to_datetime(org_group['date_code'])
        
        # 获取当前猪场的猪场ID和三级单位名称
        current_farm_dk = org_group['pigfarm_dk'].iloc[0]
        current_l3_org = org_group['l3_org_inv_nm'].iloc[0]
        
        # 计算日期跨度，用于确定图表宽度
        date_range = (org_group['date_code'].max() - org_group['date_code'].min()).days
        
        # 根据日期跨度动态调整图表宽度，确保每天都有足够空间
        figsize_width = max(48, date_range * 0.1)  # 宽度系数可以调整
        
        # 创建图形，使用动态宽度
        plt.figure(figsize=(figsize_width, figsize_height))
        
        # 排序确保日期顺序正确
        org_group = org_group.sort_values('date_code')
        
        # 准备数据
        dates = mdates.date2num(org_group['date_code'].values)  # 转换为数值类型
        abortion_rates = org_group['abortion_1_7'].values

        # 确保有足够的数据点绘制线段
        if len(dates) > 1:
            # 创建点对序列
            points = np.array([dates, abortion_rates]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # 计算每对相邻点的最大流产率值
            max_rates = np.maximum(abortion_rates[:-1], abortion_rates[1:])

            # 定义颜色和值范围
            colors = ['green', 'orange', 'red']
            bounds = [0, 0.0025, 0.005, y_max]
            
            # 创建颜色映射和规范化对象
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(bounds, cmap.N)
            
            # 根据y值为每个线段着色
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(max_rates)  # 使用起始点的值确定颜色
            lc.set_linewidth(2)
            
            # 绘图前先创建坐标轴对象
            ax = plt.gca()
            ax.add_collection(lc)
        
        # 将原始数据分为空值和非空值部分
        non_null_mask = ~np.isnan(abortion_rates)
        null_mask = np.isnan(abortion_rates)
        zero_mask = (abortion_rates == 0) & non_null_mask
        value_mask = (abortion_rates > 0) & non_null_mask
        
        # 绘制非零、非空值的点 (实心点)
        if np.any(value_mask):
            plt.scatter(dates[value_mask], abortion_rates[value_mask], s=30, color='black', zorder=5)
        
        # 绘制零值的点 (实心点)
        if np.any(zero_mask):
            plt.scatter(dates[zero_mask], abortion_rates[zero_mask], s=30, color='black', zorder=5)
        
        # 绘制空值的点 (空心点)
        if np.any(null_mask):
            plt.scatter(dates[null_mask], np.zeros_like(dates[null_mask]), s=30, facecolors='none', 
                       edgecolors='black', linewidth=1.5, zorder=5)
        
        # 添加三级单位均值线 (用蓝色线表示)
        if 'abortion_1_7_l3_mean' in org_group.columns and 'abortion_1_7_l3_var' in org_group.columns:
            l3_mean = org_group['abortion_1_7_l3_mean'].values
            l3_var = org_group['abortion_1_7_l3_var'].values
            
            # 计算上边界和下边界
            upper_bound = l3_mean + l3_var
            lower_bound = l3_mean - l3_var
            
            # 防止下边界出现负值
            lower_bound = np.maximum(lower_bound, 0)
            
            # 绘制上下边界线，使用浅紫色，线宽较细
            plt.plot(dates, upper_bound, '-', color='#8A2BE2', linewidth=1, alpha=0.7, label='_nolegend_')
            plt.plot(dates, lower_bound, '-', color='#8A2BE2', linewidth=1, alpha=0.7, label='_nolegend_')
            plt.plot(dates, l3_mean, '-', color='blue', linewidth=1.5)
            
            # 填充上下边界之间的区域，表示方差范围
            plt.fill_between(dates, lower_bound, upper_bound, color='#8A2BE2', alpha=0.1)

        if 'abortion_1_7_l3_median' in org_group.columns:
            # 添加三级单位流产率中位数线 (用紫色线表示)
            l3_median = org_group['abortion_1_7_l3_median'].values
            plt.plot(dates, l3_median, '-', color='purple', linewidth=1.5, label='_nolegend_')
        
        # 添加PRRS检测阳性率散点图 (用红色星形点表示)
        if 'prrs_check_out_ratio' in org_group.columns:
            prrs_ratios = org_group['prrs_check_out_ratio'].values
            non_null_prrs = ~np.isnan(prrs_ratios)
            if np.any(non_null_prrs):
                plt.scatter(dates[non_null_prrs], prrs_ratios[non_null_prrs], 
                           marker='*', s=150, color='red', alpha=0.7)
        
        # 添加同一三级单位其他猪场当天的流产率数据点 (透明浅灰色散点)
        other_farms_points = []
        
        # 收集所有日期和对应的三级单位其他猪场流产率值
        for date in org_group['date_code'].unique():
            # 筛选同一天同一三级单位的其他猪场
            same_day_farms = mydata[(mydata['date_code'] == date) & 
                                   (mydata['l3_org_inv_nm'] == current_l3_org) & 
                                   (mydata['pigfarm_dk'] != current_farm_dk)]
            
            # 如果有其他猪场数据
            if not same_day_farms.empty:
                # 提取这些猪场的流产率值
                for _, farm_row in same_day_farms.iterrows():
                    if pd.notna(farm_row['abortion_1_7']):
                        other_farms_points.append((mdates.date2num(date), farm_row['abortion_1_7']))
        
        # 绘制同一三级单位其他猪场的流产率散点
        if other_farms_points:
            other_x, other_y = zip(*other_farms_points)
            plt.scatter(other_x, other_y, s=15, color='gray', alpha=0.3, zorder=2, marker='o')
        
        # 添加非瘟发病日期标记
        # 筛选当前猪场的非瘟发病记录
        farm_asf_records = asf_data[asf_data['pigfarm_dk'] == current_farm_dk]
        has_asf = False
        
        # 遍历该猪场的所有非瘟发病记录
        for _, asf_record in farm_asf_records.iterrows():
            start_date = pd.to_datetime(asf_record['start_dt'])
            end_date = pd.to_datetime(asf_record['end_dt'])
            
            # 确保发病日期在图表显示范围内
            min_date = org_group['date_code'].min()
            max_date = org_group['date_code'].max()
            
            # 如果发病日期与图表时间范围有交集
            if not (end_date < min_date or start_date > max_date):
                has_asf = True
                # 调整日期范围，确保在图表范围内
                visible_start = max(start_date, min_date)
                visible_end = min(end_date, max_date)
                
                # 转换为matplotlib日期格式
                visible_start_num = mdates.date2num(visible_start)
                visible_end_num = mdates.date2num(visible_end)
                
                # 添加阴影区域标记非瘟发病期间
                plt.axvspan(visible_start_num, visible_end_num, 
                           alpha=0.2, color='red', zorder=1, 
                           label='_' if has_asf else '非瘟发病期')
                
                # 添加发病起止日期的垂直线
                plt.axvline(x=visible_start_num, color='red', linestyle='--', 
                           alpha=0.7, linewidth=1, zorder=1)
                plt.axvline(x=visible_end_num, color='red', linestyle='--', 
                           alpha=0.7, linewidth=1, zorder=1)
                
                # 添加标注，显示非瘟发病的起止日期
                plt.text(visible_start_num, y_max*0.95, start_date.strftime('%Y-%m-%d'), 
                        rotation=90, verticalalignment='top', color='red', fontsize=8)
                plt.text(visible_end_num, y_max*0.95, end_date.strftime('%Y-%m-%d'), 
                        rotation=90, verticalalignment='top', color='red', fontsize=8)
        
        # 设置x轴为日期格式
        ax = plt.gca()
        ax.xaxis_date()
        
        # 设置日期格式
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        
        # 根据数据点数量动态调整日期标签显示
        if date_range <= 60:
            # 数据点不多时，每天一个刻度
            ax.xaxis.set_major_locator(mdates.DayLocator())
        elif date_range <= 180:
            # 数据跨度半年以内，每3天一个刻度
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        elif date_range <= 365:
            # 数据跨度一年以内，每7天一个刻度
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        else:
            # 跨度超过一年，每14天一个刻度
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
        
        # 设置次要刻度为每天（不显示标签但显示刻度线）
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        
        # 固定y轴范围和刻度
        plt.ylim(0, y_max)  
        plt.yticks(np.arange(0, y_max+y_interval, y_interval))
        
        # 设置x轴范围
        plt.xlim(dates.min() - 1, dates.max() + 1)  # 增加一些边距

        # 添加水平参考线
        plt.axhline(y=0.0025, color='darkgreen', linestyle='--', alpha=0.7)
        plt.axhline(y=0.005, color='darkred', linestyle='--', alpha=0.7)
        
        # 设置x轴标签显示方式，确保可读性
        plt.xticks(rotation=90)  # 垂直显示日期标签
        plt.xlabel('日期')
        plt.ylabel('比率值')
        
        # 获取数据的开始和结束年份
        start_year = org_group['date_code'].min().year
        end_year = org_group['date_code'].max().year
        
        year_range = f"{start_year}-{end_year}" if start_year != end_year else f"{start_year}"
        plt.title(f'{year_range}年 {org_inv_nm} 流产率及相关指标变化')
        
        # 添加更新后的图例
        legend_elements = [
            Line2D([0], [0], color='black', marker='o', markersize=6, linestyle='', label='流产率值'),
            Line2D([0], [0], marker='o', markersize=6, markerfacecolor='none', markeredgecolor='black', linestyle='', label='流产率空值'),
            Line2D([0], [0], color='blue', lw=1.5, label='三级单位均值'),
            Patch(facecolor='#8A2BE2', alpha=0.1, edgecolor='#8A2BE2', linewidth=1, label='均值±方差范围'),
            Line2D([0], [0], color='red', marker='*', markersize=10, linestyle='', label='PRRS检测阳性率'),
            Line2D([0], [0], color='gray', marker='o', markersize=4, alpha=0.3, linestyle='', label='同一三级单位其他猪场'),
            Line2D([0], [0], color='red', lw=2, label='流产率 > 5‰'),
            Line2D([0], [0], color='orange', lw=2, label='2.5‰ < 流产率 ≤ 5‰'),
            Line2D([0], [0], color='green', lw=2, label='流产率 ≤ 2.5‰')
        ]
        
        # 如果有非瘟记录，添加非瘟图例
        if has_asf:
            legend_elements.append(Patch(facecolor='red', alpha=0.2, label='非瘟发病期'))
            
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(pad=3.0)  # 增加边距，确保所有标签都能显示
        
        # 保存图片，直接使用猪场名称作为文件名
        output_filename = f'{prefix}{org_inv_nm}_abortion_rate.png'
        plt.savefig(os.path.join(base_output_dir, output_filename), dpi=300, bbox_inches='tight')
        plt.close()


def draw_diagram(mydata, l2_name, asf_data):
    # 创建输出目录
    base_output_dir = f'tmp/abortion_plots/{l2_name}'
    os.makedirs(base_output_dir, exist_ok=True)

    # 提取年信息
    mydata['year'] = mydata['date_code'].dt.year

    # 画出每个猪场的所有年份流产率变化图
    draw_pigfarm_all_years(mydata, base_output_dir, asf_data)

    print(f"流产图表已保存到 {base_output_dir} 目录")
    

if __name__ == "__main__":
    # 避免中文乱码
    plt.rcParams['font.family'] = 'SimHei'
    
    # 加载非瘟发病数据
    asf_data = pd.read_csv('raw_data/ADS_PIG_FARM_AND_REARER_ONSET.csv', encoding='utf-8')
    asf_data = asf_data.rename(columns={'org_inv_dk': 'pigfarm_dk'})
    
    # 确保日期列格式正确
    asf_data['start_dt'] = pd.to_datetime(asf_data['start_dt'])
    asf_data['end_dt'] = pd.to_datetime(asf_data['end_dt'])
    
    # 加载流产率数据
    l2_name_list = ['猪业一部', '猪业二部', '猪业三部']

    for l2_name in l2_name_list:
        data = pd.read_csv(f'processed_data/ads_pig_org_total_to_ml_training_day_abortion_{l2_name}_processed.csv', encoding='utf-8')
        
        # 确保日期列名正确
        if 'stats_dt' in data.columns:
            data = data.rename(columns={'stats_dt': 'date_code'})
            
        data['date_code'] = pd.to_datetime(data['date_code'])

        # 绘制图表，传入非瘟数据
        draw_diagram(data, l2_name, asf_data)


