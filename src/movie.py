import os

if __name__ == '__main__':
    # Convert images to video
    os.system('ffmpeg -framerate 10 -i ../output/%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p ../output/mosq.mp4')
    # os.system("ffmpeg -r 1 -i ../output/%03d.png -vcodec mpeg4 -y movie.mp4")