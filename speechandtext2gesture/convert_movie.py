#白黒はっきりとさせた映像に変換させる
import cv2
import argparse
import numpy as np
import subprocess
#https://qiita.com/studio_haneya/items/fdf09768372c985a3961
#https://qiita.com/AtomJamesScott/items/ccef87b1092d7407de0d
#https://www.yutaka-note.com/entry/opencv_01#Point-3マスク画像をもとに白色部分を透明化<-Alphaがないと透明にならない
#を参考に

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', '-pr', type=str, default="pred.mp4", help='PRED MOVIE')
    parser.add_argument('--output', '-op', type=str, default="output.mp4", help='OUTPUT MOVIE')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = make_args()
    # 入力MP4ファイルのパス
    input_video_path = args.pred

    # 出力MP4ファイルのパス
    output_video_path = args.output

    # 背景として透明にしたい色（BGR形式）
    transparent_color = (255, 255, 255)  # ここを背景の色に合わせて調整

    # 動画を読み込むキャプチャオブジェクトを作成
    cap = cv2.VideoCapture(input_video_path)

    # 入力動画からフレームの幅、高さ、FPSを取得
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    # 動画を書き出すためのビデオライターを作成
    #fourcc = cv2.VideoWriter_fourcc(*'avc1')  # MP4コーデックを使用

    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4コーデックを使用
    #out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=True)
    n_frame = 0
    # 背景透過処理のメインループ
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame[frame >= 200] = 255 
        frame[frame < 200] = 0
        # 透明背景を作成
        mask = np.all(frame == transparent_color, axis=-1)
        alpha_channel = np.where(mask, 0, 255)
        # 入力フレームと透明背景を結合
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        rgba_frame[:, :, 3] = alpha_channel

        # 透明背景を含むフレームを書き出す
        #out.write(rgba_frame)
        n_frame += 1
        if n_frame % 100 == 0:
            print("Frame数: ", n_frame)
        cv2.imwrite("frame_{:04d}.png".format(n_frame), rgba_frame)
    # クリーンアップ
    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    #!echo y | ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4
    subprocess.run(['ffmpeg', '-framerate', "30", '-i', 'frame_%04d.png', '-c:v','libx264','-pix_fmt','yuv420p',output_video_path])
    subprocess.run(['rm','frame_%04d.png'])
