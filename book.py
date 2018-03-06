# coding: utf-8

import os
import sys
try:
    from StringIO import StringIO  # py2
except:
    from io import BytesIO as StringIO  # py3
from time import sleep

os.environ['CUDA_VISIBLE_DEVICES'] = ""

import requests
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from lxml import etree
from keras.models import load_model

from core import WIDTH, HEIGHT
from transform import preprocess
from experiment import parse_output

if 'model' not in locals():
    print('Loading model...')
    model = load_model(os.path.join(src, 'pure.h5'))

def getSecurityURL(html):
    img = html.xpath('//img[@id="BookingS1Form_homeCaptcha_passCode"]')
    assert len(img) == 1, 'no valid image'
    img = img[0]
    #print("Image: ", img.get('src'))
    return img.get('src')

def Image2Text(im):
    if isinstance(im, Image.Image):
        im = np.asarray(im)
    img = preprocess(im.reshape((1, HEIGHT, WIDTH, 1)))
    result = parse_output(model.predict(img))[0]
    print('Predict: ' + result)
    return result

cities = '南港 台北 板橋 桃園 新竹 苗栗 台中 彰化 雲林 嘉義 台南 左營'.split(' ')
city_map = dict( (loc, i) for i, loc in enumerate(cities))

# consider to use map for efficiency
def parseTime(t):
    hh, mm = t.split(':')
    hh, mm = int(hh), int(mm)
    ap_code = 'N' if t == '12:00' else 'P' if hh >= 12 else 'A'
    hh = 12 if hh == 0 else hh - 12 if hh >= 12 else hh
    return '%d%d%c' % (hh, mm, ap_code)

def packInfo(toDate='2018/01/01', toTime='23:30', from_='台北', to_='台中', tick_n=[0, 0, 0, 0, 1], isStudent=True, securityCode='', backDate=None, backTime=None, isBack=False):
    
    seat_base, book_base = 20, 27
    if isStudent:
        seat_base, book_base = seat_base-2, book_base-2
    
    result = {}
    result['selectStartStation'] = city_map[from_]
    result['selectDestinationStation'] = city_map[to_]
    result['trainCon:trainRadioGroup'] = '0'
    result['seatCon:seatRadioGroup'] = 'radio%d' % seat_base
    result['bookingMethod'] = 'radio%d' % book_base
    result['toTimeInputField'] = toDate
    result['toTimeTable'] = parseTime(toTime)
    result['toTrainIDInputField'] = ''
    result['backTimeInputField'] = backDate if isBack and backDate is not None else toDate
    result['backTimeTable'] = parseTime(backTime) if isBack and backTime is not None else ''
    result['backTrainIDInputField'] = ''
    for i, typ in enumerate('FHWEP'): # 全、孩童、愛心、敬老、大學生
        result['ticketPanel:rows:%d:ticketAmount'%i] = '%d%s' % (tick_n[i], typ)
    result['homeCaptcha:securityCode'] = securityCode
    result['SubmitButton'] = '開始查詢'
    return result
    
def getStatus(res_root):
    elems = res_root.xpath('//div[@id="error"]/*/ul/li/span[@class="feedbackPanelERROR"]')
    if len(elems) == 0:
        return True
    
    for elem in elems:
        print('Error: ' + elem.text)
    if not u'輸入錯誤' in ''.join([ elem.text for elem in elems]):
        return None
    return False

def packUserInfo(id_, phone_, useId=True, isMobile=True, email=''):
    return {
        'diffOver': '1',
#         'idInputRadio': 'radio33' if useId else 'radio35',
        'idInputRadio:idNumber': id_,
#         'eaiPhoneCon:phoneInputRadio': 'radio44' if isMobile else 'radio41',
        'eaiPhoneCon:phoneInputRadio:phoneNumber': phone_,
        'email': email,
        'agree': 'on',
        'isGoBackM': '',
        'backHome': ''
    }

def parse_args():
    parser = argparse.ArgumentParser('THSR booking System')
    parser.add_argument('--viz', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    if args.viz: import IPython
    from _secret import secrets

    debug = False
    startDate = '2018/03/15' if debug else '2018/03/31'
    backDate = '2018/03/17' if debug else '2018/04/08'
    isStudent = True
    isBack = False   # TODO
    baseurl = 'https://irs.thsrc.com.tw'
    users = list(map(lambda x: packUserInfo(id_=x.get('id'), phone_=x.get('phone')), secrets))
    # Adjust here to fit your needs
    data = packInfo(toDate=startDate, toTime='20:30',
            backDate=backDate, backTime='20:30', 
            from_='台南', to_='台北', 
            tick_n=[0, 0, 0, 0, len(users)],
            isStudent=isStudent, isBack=isBack)
    query = '/IMINT/?student=university' if isStudent else '/IMINT/'
    loop = True
    for userInfo in users:
        retry = 100 if loop else 1
        MAX_PASS = 100

        while retry > 0:
            try:
                counter = 0
                with requests.Session() as s:
                    res = s.get(baseurl + query)

                    # parse response HTML
                    txt = res.text
                    html = etree.HTML(res.text)
                    BookingS1Form = html.xpath('//form[@id="BookingS1Form"]')[0]
                    hidden1 = BookingS1Form.xpath('.//input[@type="hidden"]')[0].get('name')
                    action = BookingS1Form.get('action')
                    
                    a_tag = html.xpath('//a[@id="BookingS1Form_homeCaptcha_reCodeLink"]')[0]
                    href = a_tag.get('href').replace('ILinkListener', 'IBehaviorListener')
                    post_href = href.split('=')[1]

                    canPass = False
                    right_cap = False
                    while not canPass:
                        assert counter < MAX_PASS
                        if not right_cap:
                            sec_res = s.get(baseurl + getSecurityURL(html))
                            sec_image = Image.open(StringIO(sec_res.content))
                            sec_image = sec_image.convert('L').resize((WIDTH, HEIGHT), Image.BILINEAR)
                            if args.viz:  IPython.display.display(sec_image)

                            # identify the securityCode
                            securityCode = Image2Text(sec_image)
                            data[hidden1] = ''
                            data['homeCaptcha:securityCode'] = securityCode
                        
                        res_submit = s.post(url=baseurl + action, data=data)
                        res_root = etree.HTML(res_submit.text)
                        canPass = getStatus(res_root)
                    
                        if canPass is None:
                            right_cap = True
                        elif not canPass:
                            reCode = '&'.join(['wicket:interface:=' + post_href,])
                            html = etree.HTML(s.get(baseurl + '/IMINT/?' + reCode).text)
                            counter += 1

                    print('\nSecond Stage')
                    BookingS2Form = res_root.xpath('//form[@id="BookingS2Form"]')[0]
                    hidden2 = BookingS2Form.xpath('.//input[@type="hidden"]')[0].get('name')
                    if args.viz:  
                        # print table
                        df = pd.read_html(etree.tostring(BookingS2Form.xpath('.//table')[0], method='HTML'), 
                                          header=0, index_col=0)[0]
                        IPython.display.display(df.fillna(''))

                    # pick one value
                    row = BookingS2Form.xpath('//span[@id="BookingS2Form_TrainQueryDataViewPanel"]' + 
                                '/table//tr[not(@class="section_subtitle")]')[0]
                    rname = row.xpath('.//input[@name="TrainQueryDataViewPanel:TrainGroup"]')[0].get('value')
                    rname2 = ''
                    if isBack: # TODO
                        #row2 = BookingS2Form.xpath('//span[@id="BookingS2Form_TrainQueryDataViewPanel2"]' + 
                        #    '/table//tr[not(@class="section_subtitle")]')[0]
                        rname2 = BookingS2Form.xpath('.//input[@name="TrainQueryDataViewPanel2:TrainGroup"]')[0].get('value')
                    
                    action2 = BookingS2Form.get('action')
                    res_submit2 = s.post(url=baseurl + action2, data={
                        hidden2: '',
                        'TrainQueryDataViewPanel:TrainGroup': rname,
                        'TrainQueryDataViewPanel2:TrainGroup': rname2,
                        'SubmitButton': '確認車次'
                    })
                    res_root = etree.HTML(res_submit2.text)
                    assert getStatus(res_root), 'Error Occur in stage2'

                    print('\nFinal Stage')
                    if args.viz:
                        df = pd.read_html(etree.tostring(res_root.xpath('.//*[@id="content"]//table')[0], method='HTML'), 
                                          header=0)[0]
                        IPython.display.display(df.fillna('').loc[:2])
                    BookingS3FormSP = res_root.xpath('//form[@id="BookingS3FormSP"]')[0]
                    hidden3 = BookingS3FormSP.xpath('.//input[@type="hidden"]')[0].get('name')
                    action3 = BookingS3FormSP.get('action')
                    userInfo[hidden3] = ''
                    userInfo['diffOver'] = BookingS3FormSP.xpath('.//input[@name="diffOver"]')[0].get('value')
                    id_tags = BookingS3FormSP.xpath('.//input[@name="idInputRadio"]')
                    phone_tags = BookingS3FormSP.xpath('.//input[@name="eaiPhoneCon:phoneInputRadio"]')
                    userInfo['idInputRadio'] = id_tags[0].get('value')
                    userInfo['eaiPhoneCon:phoneInputRadio'] = phone_tags[0].get('value')
                    res_submit3 = s.post(url=baseurl+action3, data=userInfo)
                    res_root = etree.HTML(res_submit3.text)
                    assert getStatus(res_root), 'Error Occur in stage3'
                    orderid = res_root.xpath('//span/table[@class="table_details"]//td[@class="content_key"]/span')[0].text
                    print('orderid: ' + orderid + ', phone: ' + userInfo['eaiPhoneCon:phoneInputRadio:phoneNumber'])
            except AssertionError as e:
                #print('EXCEPT: {}'.format(e.message))
                pass
            else:
                print("Mission complete")
                break
            sleep(0.4)
            retry -= 1
