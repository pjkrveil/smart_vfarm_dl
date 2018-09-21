from shutil import copyfile
import random
import os
import sys
import cv2

# base parameter

img_path = '../dataset/'
train_path = '../dataset/train_img/'
test_path = '../dataset/test_img/'
label_path = '../dataset/label/'

class_name = ['healthy', 'threshold', 'unhealthy']
sample_name = ['sample_1', 'sample_2']

count_test = 1

ratio = 0.2


def fileRenaming(startNum, endNum, count_train, sample_num, class_num):
        global sample_name
        global class_name
        global count_test
        global ratio

        selected = []

        while len(selected) != int((endNum - startNum) * ratio):
                num = random.randrange(startNum, endNum)

                if not num in selected:
                        selected.append(num)

        print(selected)
        print(" length : %d" % len(selected))

        for i in range(startNum, endNum):
                file_num = '%003d' % i
                original_file_name = 'IMG' + file_num + '.jpg'

                if i in selected:
                        file_new_num = '%003d' % count_test
                        file_name = 'IMG' + file_new_num + '.jpg'
                        copyfile(img_path + sample_name[sample_num] + '/' + original_file_name, img_path + 'test/' + file_name)
                        count_test += 1
                else:
                        file_new_num = '%003d' % count_train
                        file_name = 'IMG' + file_new_num + '.jpg'
                        print ("")
                        print ("%s" % original_file_name)
                        copyfile(img_path + sample_name[sample_num] + '/' + original_file_name, img_path + 'train/' + class_name[class_num] + '/' + file_name)
                        count_train += 1

        return count_train
        

def dataCleanup():
        global sample_name
        global class_name

        testset1 = 1
        testset2 = 1
        testset3 = 1

        for i in range(len(sample_name)):
                class_num = 0

                #file renaming from 1 to 151
                testset1 = fileRenaming(1, 151, testset1, i, class_num)
                class_num += 1

                #file renaming from 151 to 201
                testset2 = fileRenaming(151, 201, testset2, i, class_num)
                class_num += 1

                #file renaming from 201 to 393
                testset3 = fileRenaming(201, 393, testset3, i, class_num)
                

def outdated_1():
        """ file labeling and grouping
        author: ...
        date...
        input: image files
        output: ....
        """
        
        for j in range(len(sample_name)):	
                label_csv_f = open(label_path + "/label_csv%d.txt", 'w') % j

                line = "num, label\n"
                label_csv_f.write(line)

                for i in range(392):
                        if i > 0 and i < 151:
                                data = "%d, %d\n" % i, 0
                                label_csv_f.write(data)

                        elif i > 150 and i < 201:
                                data = "%d, %d\n" % i, 1
                                label_csv_f.write(data)

                        else:
                                data = "%d, %d\n" % i, 0
                                label_csv_f.write(data)

                label_csv_f.close()

def main():
        dataCleanup()


main()
