import os
import pandas as pd
import numpy as np
import re
from shapely.geometry import LineString, Point
import glob
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')  # 'NanumBarunGothic' 폰트를 사용
plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지

log_dir = ''

logic_str = 'LogicAmkor'
#전체 분할 가로/세로 갯수
divide_size = 6
#중심제외 각 존의 셀 가로/세로 갯수
zone_size = int(divide_size / 2)
#중심 셀 가로/세로 갯수
center_size = 2
#대각선 Minmax 검사시 선 추출 공차
line_toler = 0.1

#측정방법
logic_no = 5

if logic_no == 1:
    logic_str = 'LogicAmkor6'    
elif logic_no == 2:
    logic_str = 'LogicSamsung1'
    divide_size = 10
    zone_size = 2
    center_size = 2
elif logic_no == 3:
    logic_str = 'LogicMinMax'
    divide_size = 10
    zone_size = 5
    center_size = 2
elif logic_no == 4:
    logic_str = 'LogicAmkor9'
    divide_size = 10
    zone_size = 4
    center_size = 2
elif logic_no == 5:
    logic_str = 'SemiJapan'
    divide_size = 3
    zone_size = 1
    center_size = 1

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
        warpage_diff = zone_center - warpage_avg

    return warpage_diff * 1000

# 데이터를 숫자형으로 변환하는 함수
def convert_to_numeric(column):
    return pd.to_numeric(column.str.replace(' ', ''), errors='coerce')

def AverageSection(folder_path, file):
    # 파일 경로
    file_path = os.path.join(folder_path, file)
    
    # CSV 파일 로딩
    data = pd.read_csv(file_path, names=['X', 'Y', 'Height'], skiprows=1)

    # 데이터 타입 변환
    data = data.apply(pd.to_numeric, errors='coerce')

    data_min = data['Height'].min()
    data_max = data['Height'].max()
    warpage_gap = data_max - data_min
        
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
        
    return section_avg,warpage_gap * 1000

def MinmaxSection(tray_no, unit_index, folder_path, file):
    # 파일 경로
    file_path = os.path.join(folder_path, file)
    
    # CSV 파일 로딩
    data = pd.read_csv(file_path, names=['X', 'Y', 'Height'], skiprows=1)

    # 데이터 타입 변환
    data = data.apply(pd.to_numeric, errors='coerce')
    
    data_min = data['Height'].min()
    data_max = data['Height'].max()
    warpage_gap = data_max - data_min

    cross_check = True

    warpage_diff = DiagonalSection(tray_no, unit_index, data, cross_check)
    
    return warpage_diff * 1000, warpage_gap * 1000

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


def DiagonalSection(tray_no, unit_index, data, cross_check):
    
    # 좌상단과 우하단 점 찾기
    min_x, max_x = data['X'].min(), data['X'].max()
    min_y, max_y = data['Y'].min(), data['Y'].max()

    left_top_data = data[(data['X'] >= min_x - line_toler) & (data['X'] < min_x + line_toler) &
                                (data['Y'] >= max_y - line_toler) & (data['Y'] < max_y + line_toler)]

    right_btm_data = data[(data['X'] >= max_x - line_toler) & (data['X'] < max_x + line_toler) &
                                (data['Y'] >= min_y - line_toler) & (data['Y'] < min_y + line_toler)]

    right_top_data = data[(data['X'] >= max_x - line_toler) & (data['X'] < max_x + line_toler) &
                                (data['Y'] >= max_y - line_toler) & (data['Y'] < max_y + line_toler)]

    left_btm_data = data[(data['X'] >= min_x - line_toler) & (data['X'] < min_x + line_toler) &
                                (data['Y'] >= min_y - line_toler) & (data['Y'] < min_y + line_toler)]
    
    if len(left_top_data) <= 0 or len(right_btm_data) <= 0 or len(right_top_data) <= 0 or len(left_btm_data) <= 0:
        print('failed to get diagonal profile')
        return pd.DataFrame()

    line_profiles = [pd.DataFrame() for _ in range(4)]

    
    # 대각선 방향에 대한 측정
    left_top_pt = left_top_data.values[0][:2]
    right_btm_pt = right_btm_data.values[0][:2]
    left_btm_pt = left_btm_data.values[0][:2]
    right_top_pt = right_top_data.values[0][:2]

    height_diff1 = right_btm_data.values[0][2] - left_top_data.values[0][2]
    line_profiles[0], min_val1, max_val1 = AdjustSlope(tray_no, unit_index, 0, data, left_top_pt, right_btm_pt, height_diff1)

    height_diff2 = right_top_data.values[0][2] - left_btm_data.values[0][2]
    line_profiles[1], min_val2, max_val2 = AdjustSlope(tray_no, unit_index, 1, data, left_btm_pt, right_top_pt, height_diff2)
    
    diagonal_diff = min_val1 + max_val1 + min_val2 + max_val2

    cross_diff = 0
    line_cnt  = 2

    # 십자선 방향에 대한 측정
    if cross_check:

        #십자선 4점 구하기
        left_pt = (left_top_pt + left_btm_pt) / 2
        top_pt = (left_top_pt + right_top_pt) / 2
        right_pt = (right_top_pt + right_btm_pt) / 2
        bottom_pt = (left_btm_pt + right_btm_pt) / 2

        #십자선 4점에 인접한 데이터 얻기
        left_data = data[(data['X'] >= left_pt[0] - line_toler) & (data['X'] < left_pt[0] + line_toler) &
                                (data['Y'] >= left_pt[1] - line_toler) & (data['Y'] < left_pt[1] + line_toler)]
        
        top_data = data[(data['X'] >= top_pt[0] - line_toler) & (data['X'] < top_pt[0] + line_toler) &
                                (data['Y'] >= top_pt[1] - line_toler) & (data['Y'] < top_pt[1] + line_toler)]
        
        right_data = data[(data['X'] >= right_pt[0] - line_toler) & (data['X'] < right_pt[0] + line_toler) &
                                (data['Y'] >= right_pt[1] - line_toler) & (data['Y'] < right_pt[1] + line_toler)]
        
        bottom_data = data[(data['X'] >= bottom_pt[0] - line_toler) & (data['X'] < bottom_pt[0] + line_toler) &
                                (data['Y'] >= bottom_pt[1] - line_toler) & (data['Y'] < bottom_pt[1] + line_toler)]
        
        #인접 데이터에 해당하는 XYZ값 다시 얻기
        left_pt = left_data.values[0][:2]
        top_pt = top_data.values[0][:2]
        right_pt = right_data.values[0][:2]
        bottom_pt = bottom_data.values[0][:2]
    
        height_diff3 = right_data.values[0][2] - left_data.values[0][2]        
        line_profiles[2], min_val3, max_val3 = AdjustSlope(tray_no, unit_index, 2, data, left_pt, right_pt, height_diff3)
        
        height_diff4 = bottom_data.values[0][2] - top_data.values[0][2]
        line_profiles[3], min_val4, max_val4 = AdjustSlope(tray_no, unit_index, 3, data, top_pt, bottom_pt, height_diff4)

        cross_diff = min_val3 + max_val3 + min_val4 + max_val4
        line_cnt  = 4

    chart_mode = 2

    MergeProfile(tray_no, unit_index, line_profiles)

    warpage_diff = diagonal_diff + cross_diff

    if np.isnan(warpage_diff):
        warpage_diff = 0

    return warpage_diff

def MergeProfile(tray_no, unit_index, line_profiles):
    
    line_cnt = len(line_profiles)
    
    max_pos = 0
    for i in range(line_cnt):
        
        line_max = line_profiles[i]['Dist'].max()

        if i == 0:
            max_pos = line_max
        else:
            if line_max > max_pos:
                max_pos = line_max

    for i in range(line_cnt):
        line_max = line_profiles[i]['Dist'].max()
        scale_value = max_pos / line_max
        line_profiles[i]['Dist'] = line_profiles[i]['Dist'] * scale_value
    
    max_line_dist = max_pos
    point_step = max_line_dist / 100

    # 새로운 Dist 축을 정의 (예: -0.005부터 0.005까지 0.001 간격)
    # max_pos를 포함
    new_dist = np.arange(0, max_pos + point_step, point_step)

    # 보간된 데이터프레임을 저장할 리스트 초기화 위치 조정
    profile_df = pd.DataFrame()
    profile_df['Dist'] = new_dist
    
    plt.figure(figsize=(10, 6))

    label_names = ['대각선A','대각선B','가로선','세로선']

    for i in range(line_cnt):    
        # 선형 보간
        
        line_name = 'Height' + str(i)
        profile_df[line_name] = np.interp(new_dist, line_profiles[i]['Dist'], line_profiles[i]['Height'])
        plt.plot(profile_df['Dist'], profile_df[line_name], label=label_names[i])
    
    save_name = f'tray{tray_no}_unit{unit_index}'

    # 그래프 그리기
    plt.title(save_name)
    plt.xlabel('Distance')
    plt.ylabel('Height')
    plt.legend()
    plt.grid(True)

    # 그래프 저장 (PNG 형식)
    log_image  = f'{save_name}.png'
    image_path = os.path.join(log_dir, log_image)
    plt.savefig(image_path, dpi=300)  # 파일 이름과 해상도 설정

    # plt.show()
    # plt.close()

    log_file = f'{save_name}.csv'
    log_path = os.path.join(log_dir, log_file)

    profile_df.to_csv(log_path, mode='w', index=False)


def AdjustSlope(tray_no, unit_index, line_type, data, point1, point2, height_diff):
    
    # 가상의 선 생성
    line1 = LineString([point1, point2])

    # 선에 가까운 점들 찾기
    dist_threshold = 0.5
    close_points = data[data.apply(lambda row: line1.distance(Point(row['X'], row['Y'])), axis=1) <= dist_threshold].copy()
    
    # 높이 보정    
    x_diff = point2[0] - point1[0]
    y_diff = point2[1] - point1[1]
    slope = height_diff / np.sqrt(x_diff**2 + y_diff**2)

    # 좌상단 점으로부터의 거리 계산
    close_points['Dist'] = np.sqrt((close_points['X'] - point1[0]) ** 2 + (close_points['Y'] - point1[1]) ** 2)

    close_points['Original'] = close_points['Height']
    # 높이 보정 적용
    min_idx = close_points['Dist'].idxmin()
    close_points['Height'] = close_points['Height'] - (close_points['Dist'] * slope) - close_points.loc[min_idx, 'Height']

    dist_min = close_points['Dist'].min()
    dist_max = close_points['Dist'].max()

    profile_min = 0
    profile_max = 0

    if divide_size > 0:
        dist_dist = dist_max - dist_min
        center_dist = (dist_dist / divide_size) * center_size

        center_start = (dist_dist - center_dist) * 0.5
        center_end = center_start + center_dist

        close_points['Filter'] = close_points['Height']

        close_points.loc[(close_points['Dist'] > center_start) & (close_points['Dist'] < center_end), 'Filter'] = 0
    
        profile_min = close_points['Filter'].min()
        profile_max = close_points['Filter'].max()
        
        column_names = ['X','Y','Dist','Original','Height','Filter']
        close_points = close_points[column_names]
    else:
        profile_min = close_points['Height'].min()
        profile_max = close_points['Height'].max()
        
        column_names = ['X','Y','Dist','Original','Height']
        close_points = close_points[column_names]

    close_points = close_points.sort_values(by='Dist')

    # log_file = f'tray{tray_no}_unit{unit_index}_profile{line_type}.csv'
    # log_path = os.path.join(log_dir, log_file)
    # close_points.to_csv(log_path, mode='w', index=False)

    return close_points, profile_min, profile_max

def is_starting_with_number(filename):
    # 정규표현식을 사용하여 파일명의 시작이 숫자로 이루어져 있는지 체크합니다.
    return bool(re.match(r'^\d{1}', filename))

def InspWarpageShapeTray(folder_path,folder):

    print("[Warpage shape Inspection]-",folder)

    tray_no = folder[0]

    # 폴더 내의 모든 CSV 파일을 찾기
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 결과를 저장할 빈 데이터프레임 생성
    shape_df = pd.DataFrame(columns=['Tray','Unit','Diff','Shape','Gap'])
    
    # 각 CSV 파일에 대해 반복 처리
    file_cnt = 0
    for file in csv_files:

        if not is_starting_with_number(file):
            continue
        
        unit_index = file.split('_')[0]

        warpage_gap = 0

        if logic_no == 5:            
            warpage_diff, warpage_gap = MinmaxSection(tray_no, unit_index, folder_path, file)
            if warpage_diff is None:
                return
        else:
            section_avg, warpage_gap = AverageSection(folder_path, file)
            zone_avg,zone_min,zone_max = AverageZone(section_avg)
            warpage_diff = decision_shape(zone_avg,zone_min,zone_max)
        

        warpage_shape = ''
        
        if warpage_diff > 0:
            warpage_shape = 'Smile'
        else:
            warpage_shape = 'Crying'

        shape_str = "{}-{} : {}, {:.3f}, gap {:.3f}".format(tray_no,unit_index,warpage_shape,warpage_diff,warpage_gap)
        print(shape_str)

        shape_df.loc[file_cnt] = [tray_no,unit_index,warpage_diff,warpage_shape,warpage_gap]
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
    # folder_path = r'D:\TEMP'
    # folder_path = r'D:\TEMP\Simulator'
    # folder_path = r'D:\10_고객사\01_대덕전자\대덕 A2 Unit\VOC\20240228_Warpage 알고리즘\시뮬레이션\20240404_TEST\Simulator'
    folder_path = r'D:\10_고객사\01_대덕전자\대덕 A2 Unit\VOC\20240228_Warpage 알고리즘\시뮬레이션\20240419_TEST\Simulator'
    root_path = folder_path
    folders_list = get_folders_in_directory(folder_path)

    global log_dir
    log_dir = os.path.join(folder_path, 'profile') 

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        # 폴더 내의 해당 확장자를 가진 파일 찾기
        files_to_delete = glob.glob(os.path.join(log_dir, "*.csv"))

        # 파일 삭제
        for file_path in files_to_delete:
            os.remove(file_path)
            
    # 결과를 저장할 빈 데이터프레임 생성
    final_df = pd.DataFrame()

    for folder in folders_list:
        dir_path = os.path.join(folder_path,folder)
        shape_df = InspWarpageShapeTray(dir_path,folder)
        if shape_df is None:
            print('검사 실패!')
            return

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
    merge_df['Result'] = ((merge_df['Diff'] > 0) & (merge_df['Amkor'] < 0)) | ((merge_df['Diff'] < 0) & (merge_df['Amkor'] > 0))

    total_cnt = merge_df.shape[0]
    missmatch_cnt = merge_df['Result'].sum()
    merge_df['Count'] = np.nan
    merge_df.at[0, 'Count'] = missmatch_cnt
    
    mismatch_rate = missmatch_cnt / float(total_cnt) * 100

    result_file = f'{logic_str}_Section{divide_size}_Zone{zone_size}_Center{center_size}_line{line_toler}_mismatch{mismatch_rate:.1f}.csv'
    save_path = os.path.join(folder_path, result_file)
    merge_df.to_csv(save_path,mode='w', index=False)
    # print(shape_df)

    print(merge_df.head(10))

    result_str = f"{logic_str} : 불일치 {missmatch_cnt} / 전체갯수 {total_cnt}, 비율 {mismatch_rate:.2f}%"
    print(result_str)
    print(save_path,'에 저장 완료')

if __name__ == '__main__':
    InspWarpageShapeAll()