from pytube import YouTube

YouTube('https://youtu.be/9bZkp7q19f0').streams.first().download()

yt = YouTube("https://www.youtube.com/watch?v=fkps18H3SXY")

yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()