import os
import pandas as pd
import numpy as np
import re

logic_str = 'LogicAmkor'
#전체 분할 가로/세로 갯수
divide_size = 6
#중심제외 각 존의 셀 가로/세로 갯수
zone_size = int(divide_size / 2)
#중심 셀 가로/세로 갯수
center_size = 2

#측정방법
logic_no = 1

if logic_no == 1:
    logic_str = 'LogicAmkor6'    
elif logic_no == 2:
    logic_str = 'LogicSamsung1'
    divide_size = 10
    zone_size = 2
    center_size = 2
elif logic_no == 3:
    logic_str = 'LogicMinMax'
elif logic_no == 4:
    logic_str = 'LogicAmkor9'
    divide_size = 10
    zone_size = 4
    center_size = 2
        
# warpage shape 판정
def decision_shape(zone_avg,zone_min,zone_max):

    warpage_diff = 0

    #0 : 중심, 1 : 좌상, 2:우상, 3:좌하, 4:우하
    if logic_no == 2:
        # LogicSamsung1
        zone_center = zone_avg[0]
        warpage_avg1 = (zone_avg[1] + zone_avg[4]) / 2
        warpage_avg2 = (zone_avg[2] + zone_avg[3]) / 2
        
        if abs(warpage_avg1) > abs(warpage_avg2):
            warpage_diff = warpage_avg1 - zone_center
        else:
            warpage_diff = warpage_avg2 - zone_center
    elif logic_no == 3:
        # LogicMinMax
        zone_center = zone_avg[0]

        zone4_min = zone_min[1:] - zone_center
        zone4_max = zone_max[1:] - zone_center

        warpage_diff = zone4_min.sum() + zone4_max.sum()
    elif logic_no == 4:        
        # LogicSemi
        zone_center = zone_avg[0]
        warpage_avg1 = (zone_avg[1] + zone_avg[4]) / 2
        warpage_avg2 = (zone_avg[2] + zone_avg[3]) / 2
        
        if warpage_avg1 > warpage_avg2:
            warpage_diff = warpage_avg1 - zone_center
        else:
            warpage_diff = warpage_avg2 - zone_center
    else:
        # LogicAmkor6
        warpage_avg = zone_avg[1:].mean()
        zone_center = zone_avg[0]
        warpage_diff = warpage_avg - zone_center

    return warpage_diff



# 데이터를 숫자형으로 변환하는 함수
def convert_to_numeric(column):
    return pd.to_numeric(column.str.replace(' ', ''), errors='coerce')

def AverageSection(folder_path, file):
    # 파일 경로
    file_path = os.path.join(folder_path, file)
    
    # CSV 파일 로딩
    data = pd.read_csv(file_path, names=['X', 'Y', 'Height'], header=None)
    data = data[1:]
        
    # CSV 파일 로딩, 첫 번째 행을 컬럼명으로 사용, 데이터를 숫자형으로 변환
    data['X'] = convert_to_numeric(data['X'])
    data['Y'] = convert_to_numeric(data['Y'])
    data['Height'] = convert_to_numeric(data['Height'])

    min_x, max_x = data['X'].min(), data['X'].max()
    min_y, max_y = data['Y'].min(), data['Y'].max()

    # print('bound x({}~{}),y({}~{})'.format(min_x,max_x,min_y,max_y))

    # 영역을 6등분할 크기를 계산
    x_interval = (max_x - min_x) / divide_size
    y_interval = (max_y - min_y) / divide_size

    section_avg = np.zeros(shape=(divide_size,divide_size), dtype=float)

    # 6x6 영역을 순회하며 평균 높이 계산
    for i in range(divide_size):
        for j in range(divide_size):
            # 현재 영역의 X, Y 좌표 범위를 계산
            x_start = min_x + i * x_interval
            x_end = x_start + x_interval
            y_start = min_y + j * y_interval
            y_end = y_start + y_interval
            
            # 현재 영역에 해당하는 데이터를 필터링
            area_data = data[(data['X'] >= x_start) & (data['X'] < x_end) &
                             (data['Y'] >= y_start) & (data['Y'] < y_end)]
            
            # 영역별 평균 높이를 계산하여 리스트에 추가
            avg_height = area_data['Height'].mean() if not area_data.empty else None
            section_avg[i,j] = avg_height
    return section_avg

def AverageZone(section_avg):

    center_start = int((divide_size - center_size) / 2)
    center_end = center_start + center_size - 1
        
    #0 : 중심, 1 : 좌상, 2:우상, 3:좌하, 4:우하
    zone_cnt = 5
    zone_avg = np.zeros(shape=(5), dtype=float)
    zone_min = np.zeros(shape=(5), dtype=float)
    zone_max = np.zeros(shape=(5), dtype=float)

    for k in range(zone_cnt):
        
        x_start = 0
        y_start = 0
        x_size = zone_size
        y_size = zone_size

        if k == 0:
            x_start = center_start
            y_start = center_start
            x_size = center_size
            y_size = center_size
        elif k == 1:
            x_start = 0
            y_start = 0
        elif k == 2:
            x_start = divide_size - zone_size
            y_start = 0
        elif k == 3:
            x_start = 0
            y_start = divide_size - zone_size
        elif k == 4:
            x_start = divide_size - zone_size
            y_start = divide_size - zone_size

        total_val = 0
        add_cnt = 0

        section_height = []

        # 3x3 영역을 순회하며 평균 높이 계산
        for i in range(x_size):
            for j in range(y_size):
                
                pos_x = x_start + i
                pos_y = y_start + j

                if k > 0:
                    # 중심이 아닌 Zone을 구할때, 중심 영역일때 스킵
                    if( (pos_x >= center_start and pos_x <= center_end) or
                       (pos_y >= center_start and pos_y <= center_end) ):
                       continue
                
                # print('x : ', pos_x,', y : ',pos_y)
                total_val += section_avg[pos_x,pos_y]
                add_cnt += 1

                section_height.append(section_avg[pos_x,pos_y])
        
        avg_val = total_val / add_cnt
        zone_avg[k] = avg_val
        
        if logic_no == 3:
            zone_min[k] = min(section_height)
            zone_max[k] = max(section_height)

    return zone_avg,zone_min,zone_max

def is_starting_with_number(filename):
    # 정규표현식을 사용하여 파일명의 시작이 숫자로 이루어져 있는지 체크합니다.
    return bool(re.match(r'^\d{1}', filename))

def InspWarpageShapeTray(folder_path,folder):

    print("[Warpage shape Inspection]-",folder)

    tray_no = folder[0]

    # 폴더 내의 모든 CSV 파일을 찾기
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 결과를 저장할 빈 데이터프레임 생성
    shape_df = pd.DataFrame(columns=['Tray','Unit','Diff','Shape'])
    
    # 각 CSV 파일에 대해 반복 처리
    file_cnt = 0
    for file in csv_files:

        if not is_starting_with_number(file):
            continue

        section_avg = AverageSection(folder_path, file)
        zone_avg,zone_min,zone_max = AverageZone(section_avg)
        warpage_diff = decision_shape(zone_avg,zone_min,zone_max) * 1000
            
        warpage_shape = ''
        
        if warpage_diff > 0:
            warpage_shape = 'Smile'
        else:
            warpage_shape = 'Crying'

        unit_index = file.split('_')[0]

        shape_str = "{}-{} : {}, {:.3f}".format(tray_no,unit_index,warpage_shape,warpage_diff)
        print(shape_str)

        shape_df.loc[file_cnt] = [tray_no,unit_index,warpage_diff,warpage_shape]
        file_cnt += 1

    # 'A' 컬럼을 기준으로 오름차순 정렬
    shape_df['Unit'] = shape_df['Unit'].astype(int)
    sorted_df = shape_df.sort_values(by='Unit')
    return sorted_df

def get_folders_in_directory(path):
    folders = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return folders


def InspWarpageShapeAll():

    # 폴더 경로 (예시 경로입니다, 실제 폴더 경로를 사용자가 제공해야 함)
    # folder_path = r'D:\TEMP\'
    folder_path = r'D:\TEMP'
    folders_list = get_folders_in_directory(folder_path)

    # 결과를 저장할 빈 데이터프레임 생성
    final_df = pd.DataFrame()

    for folder in folders_list:
        dir_path = os.path.join(folder_path,folder)
        shape_df = InspWarpageShapeTray(dir_path,folder)

        # 최종 결과 데이터프레임에 추가
        final_df = pd.concat([final_df, shape_df])

    final_df['Tray'] = final_df['Tray'].astype(int)
    final_df['Unit'] = final_df['Unit'].astype(int)

    # 정답 결과 통합
    result_path = os.path.join(folder_path,'AmkorResult.csv')
    result_df = pd.read_csv(result_path)
    result_df['Tray'] = result_df['Tray'].astype(int)
    result_df['Unit'] = result_df['Unit'].astype(int)

    merge_df = pd.merge(final_df,result_df,on=['Tray','Unit'], how='left')
    merge_df['Diff'] = merge_df['Diff']
    merge_df['Result'] = ((merge_df['Diff'] >= 0) & (merge_df['Amkor'] >= 0)) | ((merge_df['Diff'] < 0) & (merge_df['Amkor'] < 0))

    match_cnt = merge_df['Result'].sum()
    merge_df['Count'] = np.nan    
    merge_df.at[0, 'Count'] = match_cnt
    total_cnt = merge_df.shape[0]
    mismatch_rate = match_cnt / float(total_cnt) * 100

    result_file = f'{logic_str}_Section{divide_size}_Zone{zone_size}_Center{center_size}_mismatch{mismatch_rate:.1f}.csv'
    save_path = os.path.join(folder_path, result_file)
    merge_df.to_csv(save_path,mode='w', index=False)
    # print(shape_df)

    print(merge_df.head(10))

    result_str = f"불일치 {match_cnt} / 전체갯수 {total_cnt}, 비율 {mismatch_rate:.2f}%"
    print(result_str)
    print(save_path,'에 저장 완료')

if __name__ == '__main__':
    InspWarpageShapeAll()