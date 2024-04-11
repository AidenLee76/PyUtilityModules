import os
import pandas as pd
import numpy as np
import re

divide_size = 6
zone_size = int(divide_size / 2)

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

    #0 : 중심, 1 : 좌상, 2:우상, 3:좌하, 4:우하
    zone_cnt = 5
    zone_avg = np.zeros(shape=(5), dtype=float)

    for k in range(zone_cnt):
        
        x_start = 0
        y_start = 0
        x_size = zone_size
        y_size = zone_size

        if k == 0:
            x_start = 2
            y_start = 2
            x_size = 2
            y_size = 2
        elif k == 1:
            x_start = 0
            y_start = 0
        elif k == 2:
            x_start = 3
            y_start = 0
        elif k == 3:
            x_start = 0
            y_start = 3
        elif k == 4:
            x_start = 3
            y_start = 3

        total_val = 0
        add_cnt = 0

        # 3x3 영역을 순회하며 평균 높이 계산
        for i in range(x_size):
            for j in range(y_size):
                
                pos_x = x_start + i
                pos_y = y_start + j

                if k > 0:
                    # 중심이 아닌 Zone을 구할때, 중심 영역일때 스킵
                    if((pos_x == 2 or pos_x == 3) and (pos_y == 2 or pos_y == 3)):
                       continue
                
                # print('x : ', pos_x,', y : ',pos_y)

                total_val += section_avg[pos_x,pos_y]
                add_cnt += 1
        
        avg_val = total_val / add_cnt
        zone_avg[k] = avg_val

    return zone_avg

# warpage shape 판정
def decision_shape(zone_avg):
    warpage_avg = zone_avg[1:].mean()
    zone_center = zone_avg[0]
    warpage_diff = warpage_avg - zone_center

    return warpage_diff


def is_starting_with_number(filename):
    # 정규표현식을 사용하여 파일명의 시작이 숫자로 이루어져 있는지 체크합니다.
    return bool(re.match(r'^\d{1}', filename))

def InspWarpageShape(folder_path,folder):

    print("[Warpage shape Inspection]-",folder)

    # 폴더 내의 모든 CSV 파일을 찾기
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 결과를 저장할 빈 데이터프레임 생성
    shape_df = pd.DataFrame(columns=['Unit','Shape','Diff'])
    
    # 각 CSV 파일에 대해 반복 처리
    file_cnt = 0
    for file in csv_files:

        if not is_starting_with_number(file):
            continue

        section_avg = AverageSection(folder_path, file)
        zone_avg = AverageZone(section_avg)
        warpage_diff = decision_shape(zone_avg) * 1000
            
        warpage_shape = ''
        
        if warpage_diff > 0:
            warpage_shape = 'Smile'
        else:
            warpage_shape = 'Crying'

        unit_index = file.split('_')[0]

        shape_str = "{} : {}, {:.3f}um".format(unit_index,warpage_shape,warpage_diff)
        print(shape_str)

        shape_df.loc[file_cnt] = [unit_index,warpage_shape,warpage_diff]
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

# 폴더 경로 (예시 경로입니다, 실제 폴더 경로를 사용자가 제공해야 함)
folder_path = r'D:\TEMP'
folders_list = get_folders_in_directory(folder_path)

for folder in folders_list:
    dir_path = os.path.join(folder_path,folder)
    shape_df = InspWarpageShape(dir_path,folder)
    save_path = os.path.join(folder_path,'shape_' + folder + '.csv')
    shape_df.to_csv(save_path,mode='w', index=False)
    # print(shape_df)
    print(save_path,'에 저장')

print('완료')