import time
import cv2
import numpy as np
from scipy import signal
import librosa
import pywt

class VideoTransformer:
    def __init__(self):
        self.kernel = np.array([[1, 1, 1],
                              [1, -8, 1],
                              [1, 1, 1]])
        
    def transform_frame(self, frame):
        # 1. 非线性变换 - 使用拉普拉斯算子扰动纹理特征
        laplacian = cv2.filter2D(frame, -1, self.kernel)
        frame = cv2.addWeighted(frame, 0.7, laplacian, 0.3, 0)
        
        # 2. 几何变形 - 使用波形扭曲
        rows, cols = frame.shape[:2]
        map_x = np.zeros((rows,cols), np.float32)
        map_y = np.zeros((rows,cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                map_x[i,j] = j + 3*np.sin(i/30)
                map_y[i,j] = i + 3*np.sin(j/30)
        frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        
        # 3. 纹理重构 - 使用小波变换
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y = frame_yuv[:,:,0]
        coeffs = pywt.dwt2(y, 'haar')
        # 修改小波系数
        cA, (cH, cV, cD) = coeffs
        cA = cA * 1.1
        y_rec = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        frame_yuv[:,:,0] = y_rec
        frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
        
        return frame
    
    def transform_frame_subtle(self, frame):
        # 1. 非线性变换 - 使用拉普拉斯算子扰动纹理特征
        laplacian = cv2.filter2D(frame, -1, self.kernel)
        # 减少拉普拉斯算子的影响
        frame = cv2.addWeighted(frame, 0.9, laplacian, 0.1, 0)
        
        # 2. 几何变形 - 使用波形扭曲
        rows, cols = frame.shape[:2]
        map_x = np.zeros((rows, cols), np.float32)
        map_y = np.zeros((rows, cols), np.float32)
        # 减少波形扭曲的幅度
        for i in range(rows):
            for j in range(cols):
                map_x[i, j] = j + 1.5 * np.sin(i / 60)
                map_y[i, j] = i + 1.5 * np.sin(j / 60)
        frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        
        # 3. 纹理重构 - 使用小波变换
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y = frame_yuv[:, :, 0]
        coeffs = pywt.dwt2(y, 'haar')
        # 减少小波系数的修改
        cA, (cH, cV, cD) = coeffs
        cA = cA * 1.05
        y_rec = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        frame_yuv[:, :, 0] = y_rec
        frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
        
        return frame

    def transform_audio(self, audio):
        # 4. 音频变换 - 使用相位声码器
        D = librosa.stft(audio)
        D_harmonic = librosa.decompose.hpss(D)[0]
        audio_harm = librosa.istft(D_harmonic)
        return audio_harm

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        # 读取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 处理帧
            processed_frame = self.transform_frame(frame)
            out.write(processed_frame)
            
        cap.release()
        out.release()

    def process_image(self, input_path, output_path):
        # 读取图片
        frame = cv2.imread(input_path)
        if frame is None:
            raise ValueError(f"无法读取图片: {input_path}")
            
        # 处理图片
        # cpu处理
        # processed_frame = self.transform_frame(frame)
        # cpu处理 微小变化
        processed_frame = self.transform_frame_subtle(frame)
        
        # 保存处理后的图片
        cv2.imwrite(output_path, processed_frame)
        
        
        
transformer = VideoTransformer()
startTime = time.time()
transformer.process_image('test.png', 'test_processed.png')
endTime = time.time()
print(f"处理图片时间: {endTime - startTime}秒")


# import torch
# print(torch.version.cuda)