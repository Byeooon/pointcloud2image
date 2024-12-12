import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

def load_calibration(calib_path):
    """Calibration 파일을 로드하고 필요한 행렬을 반환."""
    with open(calib_path, 'r') as f:
        calib = f.readlines()
    P2 = np.array([float(x) for x in calib[2].strip().split(' ')[1:]]).reshape(3, 4)
    R0_rect = np.array([float(x) for x in calib[4].strip().split(' ')[1:]]).reshape(3, 3)
    Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip().split(' ')[1:]]).reshape(3, 4)

    # R0_rect 확장 (3x3 -> 4x4)
    R0_rect = np.pad(R0_rect, ((0, 1), (0, 1)), mode='constant')
    R0_rect[3, 3] = 1

    # Tr_velo_to_cam 확장 (3x4 -> 4x4)
    Tr_velo_to_cam = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))

    return P2, R0_rect, Tr_velo_to_cam

def load_point_cloud(bin_path):
    """Point cloud 데이터를 로드."""
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return scan[:, :3]

def filter_points(points):
    """카메라 후방과 z 축 음수인 포인트 제거."""
    # 동차 좌표계로 변환
    velo = np.hstack((points, np.ones((points.shape[0], 1)))).T
    # 카메라 후방(x < 0) 포인트 제거
    forward_indices = velo[0, :] >= 0
    velo = velo[:, forward_indices]
    return velo

def project_points(velo, P2, R0_rect, Tr_velo_to_cam):
    """포인트들을 이미지 평면으로 투영."""
    cam_coords = P2 @ R0_rect @ Tr_velo_to_cam @ velo
    # z 축이 양수인 포인트만 사용
    valid_indices = cam_coords[2, :] > 0
    cam_coords = cam_coords[:, valid_indices]
    # u, v 좌표 정규화
    cam_coords[:2, :] /= cam_coords[2, :]
    return cam_coords

def filter_outliers_and_distance(cam_coords, img_width, img_height, max_distance=25):
    """이미지 범위 및 전방 거리 내 포인트 필터링."""
    u, v, z = cam_coords[0, :], cam_coords[1, :], cam_coords[2, :]
    # 이미지 범위를 벗어나는 포인트 제거
    valid_img = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    # 전방 거리 제한
    valid_distance = z <= max_distance
    # 두 조건을 모두 만족하는 포인트 선택
    valid_indices = valid_img & valid_distance
    return cam_coords[:, valid_indices]

def visualize_points(img, cam_coords, output_path, name):
    """포인트들을 이미지 위에 시각화하고 저장."""
    plt.figure(figsize=(12, 5), dpi=96, tight_layout=True)
    plt.axis([0, img.shape[1], img.shape[0], 0])
    plt.imshow(img)

    u, v, z = cam_coords
    plt.scatter(u, v, c=z, cmap='rainbow_r', alpha=0.5, s=2)
    plt.title(name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

def main():
    sn = int(sys.argv[1]) if len(sys.argv) > 1 else 7  # 샘플 번호 (0-7517)
    name = f'{sn:06d}'  # 6자리 0-padding

    # 파일 경로 설정
    img_path = f'./image_data/{name}.png'
    bin_path = f'./velodyne/{name}.bin'
    calib_path = f'./calib/{name}.txt'
    output_path = f'./projection_output/{name}.png'

    # 데이터 로드
    P2, R0_rect, Tr_velo_to_cam = load_calibration(calib_path)
    points = load_point_cloud(bin_path)
    velo = filter_points(points)
    cam_coords = project_points(velo, P2, R0_rect, Tr_velo_to_cam)

    # 이미지 로드 및 크기 가져오기
    img = mpimg.imread(img_path)
    IMG_H, IMG_W, _ = img.shape

    # 이미지 범위 및 전방 거리 제한 필터링
    cam_coords = filter_outliers_and_distance(cam_coords, IMG_W, IMG_H, max_distance=25)

    # 시각화 및 저장
    visualize_points(img, cam_coords, output_path, name)

if __name__ == '__main__':
    main()
