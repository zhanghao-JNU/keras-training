#-*- coding: UTF-8 -*- 

'''
1、读取指定目录下的所有文件
2、读取指定文件，输出文件内容
3、创建一个文件并保存到指定目录
'''
#coding:utf8
import os
import cv2.cv as cv
import cv2
# 遍历指定目录，显示目录下的所有文件名
width_scale = 128
height_scale = 128
write_path = "/home/zhanghao/data/classification/train_scale/"
def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath,allDir))
	write_child = os.path.join('%s%s' % (write_path,allDir))
	image = cv.LoadImage(child,0)
	des_image = cv.CreateImage((width_scale,height_scale),image.depth,1)
	cv.Resize(image,des_image,cv2.INTER_AREA)
#	cv.ShowImage('afe',des_image)
	cv.SaveImage(write_child,des_image)
#	break
#       print child.decode('gbk') # .decode('gbk')是解决中文显示乱码问题

# 读取文件内容并打印
def readFile(filename):
    fopen = open(filename, 'r') # r 代表read
    for eachLine in fopen:
        print "读取到得内容如下：",eachLine
    fopen.close()
    
# 输入多行文字，写入指定文件并保存到指定文件夹
def writeFile(filename):
    fopen = open(filename, 'w')
    print "\r请任意输入多行文字"," ( 输入 .号回车保存)"
    while True:
        aLine = raw_input()
        if aLine != ".":
            fopen.write('%s%s' % (aLine, os.linesep))
        else:
            print "文件已保存!"
            break
    fopen.close()

if __name__ == '__main__':
#     filePath = "/home/zhanghao/data/classification/1.txt"
#     filePathI = "/home/zhanghao/data/classification/2.py"
     filePathC = "/home/zhanghao/data/classification/train/"
     eachFile(filePathC)
     cv.WaitKey(0)
#     readFile(filePath)
#     writeFile(filePathI)
