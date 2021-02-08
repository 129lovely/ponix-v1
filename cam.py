
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, 'src')
import argparse
import numpy as np
import transform, vgg, pdb, os
import tensorflow as tf
import cv2
from datetime import date
from datetime import datetime
## 추가 
import os 
import easydict
from PIL import Image, ImageFont, ImageDraw 

date_obj = date.today()
year = date_obj.year
month = date_obj.month
day = date_obj.day


models=[
	{"ckpt":"models/ckpt_cubist_b20_e4_cw05/fns.ckpt", "style":"styles/cubist-landscape-justineivu-geanina.jpg"}, 	# 1
	{"ckpt":"models/ckpt_hokusai_b20_e4_cw15/fns.ckpt", "style":"styles/hokusai.jpg"},							  	# 2
	{"ckpt":"models/ckpt_kandinsky_b20_e4_cw05/fns.ckpt", "style":"styles/kandinsky2.jpg"},							# 3
	{"ckpt":"models/ckpt_liechtenstein_b20_e4_cw15/fns.ckpt", "style":"styles/liechtenstein.jpg"},					# 4
	{"ckpt":"models/ckpt_wu_b20_e4_cw15/fns.ckpt", "style":"styles/wu4.jpg"},										# 5
	{"ckpt":"models/ckpt_elsalahi_b20_e4_cw05/fns.ckpt", "style":"styles/elsalahi2.jpg"},							# 6
	{"ckpt":"models/ckpt_udnie/udnie.ckpt", "style":"styles/udnie.jpg"},											# 7
	{"ckpt":"models/ckpt_maps3_b5_e2_cw10_tv1_02/fns.ckpt", "style":"styles/maps3.jpg"},							# 8

	{"ckpt":"models/ckpt_kandinsky/fns.ckpt", "style":"styles/kandinsky.jpg"},										# 9
	{"ckpt":"models/ckpt_scream/scream.ckpt", "style":"styles/the_scream.jpg"},										# 10
	{"ckpt":"models/ckpt_rain_princess/rain_princess.ckpt", "style":"styles/rain_princess.jpg"},					# 11
	{"ckpt":"models/ckpt_la_muse/la_muse.ckpt", "style":"styles/la_muse.jpg"},										# 12
	{"ckpt":"models/ckpt_wave/wave.ckpt", "style":"styles/wave.jpg"},												# 13
	{"ckpt":"models/ckpt_croquis/fns.ckpt", "style":"styles/croquis.png"}											# 14
	]


# # parser
# parser = argparse.ArgumentParser()
# parser.add_argument('--device_id', type=int, help='camera device id (default 0)', required=False, default=0)
# # 카메라 ID
# parser.add_argument('--width', type=int, help='width to resize camera feed to (default 320)', required=False, default=640)
# # 화질
# parser.add_argument('--disp_width', type=int, help='width to display output (default 640)', required=False, default=1200)
# # display 출력 크기
# parser.add_argument('--disp_source', type=int, help='whether to display content and style images next to output, default 1', required=False, default=1)
# # 스타일 변경 전 후 보여주기 / 변경된 스타일만 보여주기
# parser.add_argument('--horizontal', type=int, help='whether to concatenate horizontally (1) or vertically(0)', required=False, default=1)
# # 가로 세로 조정
# parser.add_argument('--num_sec', type=int, help='number of seconds to hold current model before going to next (-1 to disable)', required=False, default=3)
# # 자동 전환 (초당)

# pr_width 추가 
opts = easydict.EasyDict({
    "device_id":0,
    "width":700,
    "disp_width":1250,
    "disp_source":1,
    "horizontal":1,
    "num_sec":10,
    "pr_width":10
})


# In[2]:


def load_checkpoint(checkpoint, sess):
	saver = tf.train.Saver()
	try:
		saver.restore(sess, checkpoint)
		# print('[load_checkpoint] checkpoint: ', checkpoint)
		style = cv2.imread(checkpoint)
		return True
	except:
		print("checkpoint %s not loaded correctly" % checkpoint)
		return False


def get_camera_shape(cam):
	""" use a different syntax to get video size in OpenCV 2 and OpenCV 3 """
	cv_version_major, _, _ = cv2.__version__.split('.')
	if cv_version_major == '3':
		return cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
	else:
		return cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)


def make_triptych(disp_width, frame, style, output, horizontal=True):
	print('in make_triptych')
	ch, cw, _ = frame.shape
	sh, sw, _ = style.shape
	oh, ow, _ = output.shape
	# print('output: ', output)
	# print('oh: ', oh, 'ow: ', ow)
	disp_height = int(disp_width * oh / ow)
	h = int(ch * disp_width * 0.5 / cw)
	w = int(cw * disp_height * 0.5 / ch)
	if horizontal:
		full_img = np.concatenate([
			cv2.resize(frame, (int(w), int(0.5*disp_height))),
			cv2.resize(style, (int(w), int(0.5*disp_height)))], axis=0)
		# print('disp_width: ', disp_width, 'disp_height: ', disp_height)
		###################### modify ############################
		# disp_height = disp_height
		# disp_width = disp_width
		full_img = np.concatenate([full_img, cv2.resize(output, (disp_width, disp_height))], axis=1)
	else:
		full_img = np.concatenate([
			cv2.resize(frame, (int(0.5 * disp_width), h)),
			cv2.resize(style, (int(0.5 * disp_width), h))], axis=1)
		full_img = np.concatenate([full_img, cv2.resize(output, (disp_width, disp_width * oh // ow))], axis=0)
	return full_img


# In[ ]:


def main(device_id, width, disp_width, disp_source, horizontal, num_sec, pr_width):
    t1 = datetime.now()
    idx_model = 0
    count=0

    device_t='/gpu:0'
    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    fontpath='/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf'
    font= ImageFont.truetype(fontpath, 150)

    with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess:
        cam = cv2.VideoCapture(device_id)
        # (device_id 번 째) 카메라 오브젝트 생성

        cv2.namedWindow("PONIX", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("PONIX", cv2.WND_PROP_FULLSCREEN, 1)

        cam_width, cam_height = get_camera_shape(cam)

        # print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print('cam_width: ', cam_width, 'cam_height: ',  cam_height)


        width = width if width % 4 == 0 else width + 4 - (width % 4) # must be divisible by 4

        height = int(width * float(cam_height/cam_width))
        height = height if height % 4 == 0 else height + 4 - (height % 4) # must be divisible by 4

        img_shape = (height, width, 3)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print('img_shape: ', img_shape)
        #
        batch_shape = (1,) + img_shape
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print('batch_shape: ', batch_shape)

        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
        preds = transform.net(img_placeholder)

        # load checkpoint
        load_checkpoint(models[idx_model]["ckpt"], sess)

        style = cv2.imread(models[idx_model]["style"])
        # print('[main] style: ', style)
        
        #### 파일 저장을 위해 추가 
        nm = 0
        # enter cam loop
        while True:
            ret, frame = cam.read()
            # ret = 이미지 가져오면 true / frame  카메라에서 가져온 이미지
            print('width: ', width, 'height: ', height)
            frame = cv2.resize(frame, (width, height))
            frame = cv2.flip(frame, 1)
            # 1은 좌우 반전, 0은 상하 반전
            X = np.zeros(batch_shape, dtype=np.float32)
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!')
            # print('X: ', X) # X에는 이미지들 있음
            # print('X.shape: ', X.shape)
            X[0] = frame
            # print('X[0].shape: ', X[0].shape)
            # print('X.shape: ', X.shape)

            output = sess.run(preds, feed_dict={img_placeholder:X})
            output = output[:, :, :, [2,1,0]].reshape(img_shape)
            output = np.clip(output, 0, 255).astype(np.uint8)
            output = cv2.resize(output, (width, height))

            # print('width: ', width, 'height: ', height, 'disp_width: ', disp_width)
            # 1280, 960

            # 출력 화면 조정
            if disp_source:
                full_img = make_triptych(disp_width, frame, style, output, horizontal)
                cv2.imshow('frame', full_img)
            else:
                print('not disp_source ')
                oh, ow, _ = output.shape
                output = cv2.resize(output, (disp_width, int(oh * disp_width / ow)))
                cv2.imshow('frame', output)

            ######################################################################################
            ##-------------------------    추가 기능    -----------------------------------------##
            ######################################################################################
            key_ = cv2.waitKey(1) 
            # 끝내기 esc (loop에서 빠져나오기) 
            if key_ == 27:
                break
            #화면 멈춤 stop
            elif key_ == ord('s'):
                while True:
                    key2 = cv2.waitKey(1) 
                    cv2.imshow('frame', full_img)
                                       
                    if key2 == ord('s'):
                        break
                    
                    # stop before
                    elif key2 == ord('b'):
                        idx_model = (idx_model + len(models) - 1) % len(models)
                        # print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
                        load_checkpoint(models[idx_model]["ckpt"], sess)
                        style = cv2.imread(models[idx_model]["style"])
                        
                        output = sess.run(preds, feed_dict={img_placeholder:X})
                        output = output[:, :, :, [2,1,0]].reshape(img_shape)
                        output = np.clip(output, 0, 255).astype(np.uint8)
                        output = cv2.resize(output, (width, height)) 
                        
                        full_img = make_triptych(disp_width, frame, style, output, horizontal)
                        cv2.imshow('frame',full_img)
                        
                    # stop after
                    elif key2 == ord('a'):
                        idx_model = (idx_model + 1) % len(models)
                        # print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
                        load_checkpoint(models[idx_model]["ckpt"], sess)
                        style = cv2.imread(models[idx_model]["style"])
                                
                        output = sess.run(preds, feed_dict={img_placeholder:X})
                        output = output[:, :, :, [2,1,0]].reshape(img_shape)
                        output = np.clip(output, 0, 255).astype(np.uint8)
                        output = cv2.resize(output, (width, height))
                        
                        full_img = make_triptych(disp_width, frame, style, output, horizontal)
                        cv2.imshow('frame',full_img)
                    # stop print
                    
                    elif key2 == ord('p'):
                        # resizing
                        img = cv2.resize(output,(5120, 3840))
                        cv2.imwrite('./print' + '/' 'print.png', img)
                        os.system("lpr ./print/print.png")
                    elif key2 == ord('o'):
                        # resizing
                        pr_img = np.zeros((3840,5120,3),dtype='uint8')+255
                        resizeH = int(3840*1)
                        img = cv2.resize(output,(5120, resizeH))
                        pr_img[:resizeH,:,:] = img
                        img_pil = Image.fromarray(pr_img)
                        draw = ImageDraw.Draw(img_pil)
                        #draw.text((640,3640), '포항공과대학교 인공지능대학원·연구원 방문 기념  2020.09.11', font=font, fill=(255,255,255,0))
                        draw.text((640,3640), '포항공과대학교 인공지능연구원 방문 기념  {}.{}.{}'.format(year, month, day), font=font, fill=(255,255,255,0))
                        pr_img = np.array(img_pil)
                        cv2.imwrite('./print' + '/' 'print.png', pr_img)
                        #cv2.imwrite('./print' + '/' 'print_frame{}.png'.format(count), frame)
                        #count+=1
                        os.system("lpr ./print/print.png")
                        
                        
            # 다음 테마로 가기 after
            elif key_ == ord('a'):
                idx_model = (idx_model + 1) % len(models)
                # print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
                load_checkpoint(models[idx_model]["ckpt"], sess)
                style = cv2.imread(models[idx_model]["style"])
                        
                 
            # 해당 화면의 결과 캡처 및 저장 capture
            elif key_ == ord('c'):
                cv2.imwrite('./capture' + '/' + './capture_%s.png'%nm , output)
                print("picture is saved!")
                nm +=1
               
            # 이전 테마로 되돌아가기 before
            elif key_ == ord('b'):
                idx_model = (idx_model + len(models) - 1) % len(models)
                # print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
                load_checkpoint(models[idx_model]["ckpt"], sess)
                style = cv2.imread(models[idx_model]["style"])

            # 다음 테마로 가기 after
            elif key_ == ord('a'):
                idx_model = (idx_model + 1) % len(models)
                # print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
                load_checkpoint(models[idx_model]["ckpt"], sess)
                style = cv2.imread(models[idx_model]["style"])
            # 프린트 prints

            elif key_ == ord('p'):
                # 용지 크기에 맞게 resizing
                pr_img = cv2.resize(output,(5120,3840))
                cv2.imwrite('./print' + '/' 'print.png', pr_img)
                #os.system("lpr ./print/print.png")

            elif key_ == ord('o'):
                # resizing
                pr_img = np.zeros((3840,5120,3),dtype='uint8')+255
                resizeH = int(3840*1)
                img = cv2.resize(output,(5120, resizeH))
                pr_img[:resizeH,:,:] = img
                img_pil = Image.fromarray(pr_img)
                draw = ImageDraw.Draw(img_pil)
                #draw.text((640,3640), '포항공과대학교 인공지능대학원·연구원 방문 기념  2020.09.11', font=font, fill=(255,255,255,0))
                draw.text((640,3640), '포항공과대학교 인공지능연구원 방문 기념  {}.{}.{}'.format(year, month, day), font=font, fill=(255,255,255,0))
                pr_img = np.array(img_pil)
                cv2.imwrite('./print' + '/' 'print.png', pr_img)
                #cv2.imwrite('./print' + '/' 'print_frame{}.png'.format(count), frame)
                #count+=1
                os.system("lpr ./print/print.png")
           

            t2 = datetime.now()
            dt = t2-t1

            if num_sec> 0 and dt.seconds > num_sec:
                t1 = datetime.now()
                idx_model = (idx_model + 1) % len(models)
                # print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
                load_checkpoint(models[idx_model]["ckpt"], sess)
                style = cv2.imread(models[idx_model]["style"])

                
        # done
        cam.release()
        cv2.destroyAllWindows()
        # 화면에 나타난 거 종료


# In[ ]:


if __name__ == '__main__':
	main(opts.device_id, opts.width, opts.disp_width, opts.disp_source==1, opts.horizontal==1, opts.num_sec, opts.pr_width),

