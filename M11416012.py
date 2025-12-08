import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 步驟 1: 設定檔案名稱 ---
# 在 VS Code 中，請直接將 'go.jpg' 拖曳到與程式碼相同的資料夾中
filename = 'go.jpg'

# --- 步驟 2: 檢查檔案是否存在 (防呆機制) ---
if not os.path.exists(filename):
    print(f"錯誤: 找不到檔案 '{filename}'")
    print("請確認您已將圖片拖曳到 VS Code 左側的檔案列表，且名稱相符。")
else:
    print(f"已讀取檔案: {filename}")
    
    # --- 步驟 3: 圖像處理 ---
    img = cv2.imread(filename)

    if img is None:
        print("錯誤: 雖然檔案存在，但 OpenCV 無法讀取 (可能是格式不支援或路徑問題)。")
    else:
        # 轉灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Sobel 邊緣偵測
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        
        # 將 Sobel 正規化並轉為 8-bit，方便顯示
        sobel_uint8 = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
        sobel_uint8 = sobel_uint8.astype(np.uint8)

        # Canny 邊緣偵測
        canny = cv2.Canny(gray, 100, 200)

        # 設定顯示標題與圖片
        titles = ['Original', 'Gray', 'Sobel', 'Canny']
        # OpenCV 讀入為 BGR，顯示需轉為 RGB
        # 注意：Sobel 必須轉為 uint8，否則 matplotlib 無法正確顯示
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), gray, sobel_uint8, canny]

        # --- 步驟 4: 繪圖呈現 ---
        plt.figure(figsize=(12, 8))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            img_i = images[i]
            # 根據影像維度決定是否使用灰階 colormap
            if img_i.ndim == 2:
                plt.imshow(img_i, cmap='gray')
            else:
                plt.imshow(img_i)
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()