# coding: utf-8

import os
import time

import argparse
import pandas as pd

import config
import interface
from interface import booking, cancel_book

class Book:
    def __init__(self):
        from _secret import secrets
        self.users = secrets
        assert len(secrets) > 0, "Cannot book without passenger information"
        self.file = 'ticket_list.txt'
        if os.path.exists(self.file):
            self.orders = pd.read_csv(self.file, dtype='str')
        else:
            self.orders = pd.DataFrame([], columns=['orderId', 'phone'], dtype='str')
        self.init_cnt = len(self.orders)
        self.counter = 0

    @property
    def uid(self):
        return self.users[0]['uid']

    @property
    def phone(self):
        return self.users[0]['phone']

    def add(self, orderid):
        self.orders = self.orders.append({'orderId': orderid, 'phone': self.phone}, ignore_index=True)
        self.counter += 1
        if self.counter > 3:
            self.write()

    def cancel(self, orderid):
        try:
            cancel_book(self.uid, orderid)
            self.orders = self.orders[self.orders.orderId != orderid]
        except AssertionError as e:
            print('Ignore...')
        else:
            print('Cancel %s success' % orderid)

    def cancelAll(self):
        for oid in list(self.orders['orderId']):
            self.cancel(oid)

    def write(self):
        self.orders.to_csv(self.file, index=False)
        self.counter = 0


cities = '南港 台北 板橋 桃園 新竹 苗栗 台中 彰化 雲林 嘉義 台南 左營'.split(' ')
city_map = dict( (loc, i) for i, loc in enumerate(cities))

# consider to use map for efficiency
def parseTime(t):
    hh, mm = t.split(':')
    hh, mm = int(hh), int(mm)
    ap_code = 'N' if t == '12:00' else 'P' if hh >= 12 else 'A'
    hh = 12 if hh == 0 else hh - 12 if hh >= 12 else hh
    return '%d%d%c' % (hh, mm, ap_code)


def packInfo(toDate='2018/01/01', toTime='23:30', from_='台北', to_='台中', tick_n=[0, 0, 0, 0, 1], isStudent=True, backDate=None, backTime=None, incBack=False):
    
    # seat_base, book_base = 20, 27
    # if isStudent:
    #     seat_base, book_base = seat_base-2, book_base-2
    
    result = {}
    result['selectStartStation'] = city_map[from_]
    result['selectDestinationStation'] = city_map[to_]
    result['trainCon:trainRadioGroup'] = '0'
    #result['seatCon:seatRadioGroup'] = 'radio%d' % seat_base
    #result['bookingMethod'] = 'radio%d' % book_base
    result['toTimeInputField'] = toDate
    result['toTimeTable'] = parseTime(toTime)
    result['toTrainIDInputField'] = ''
    result['backTimeCheckBox'] = 'on' if incBack else ''
    result['backTimeInputField'] = backDate if incBack and backDate is not None else toDate
    result['backTimeTable'] = parseTime(backTime) if incBack and backTime is not None else ''
    result['backTrainIDInputField'] = ''
    for i, typ in enumerate('FHWEP'): # 全、孩童、愛心、敬老、大學生
        result['ticketPanel:rows:%d:ticketAmount'%i] = '%d%s' % (tick_n[i], typ)
    result['SubmitButton'] = '開始查詢'
    return result
    

def packUserInfo(uid, phone, useId=True, isMobile=True, email=''):
    return {
        'diffOver': '1',
#         'idInputRadio': 'radio33' if useId else 'radio35',
        'idInputRadio:idNumber': uid,
#         'eaiPhoneCon:phoneInputRadio': 'radio44' if isMobile else 'radio41',
        'eaiPhoneCon:phoneInputRadio:phoneNumber': phone,
        'email': email,
        'agree': 'on',
        'isGoBackM': '',
        'backHome': ''
    }

def parse_args():
    parser = argparse.ArgumentParser('THSR booking System')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--cancel', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':

    config.args = args = parse_args()

    if 'model' not in locals():
        # specify before import tensorflow/keras to use CPU to predict
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        from keras.models import load_model
        print('Loading model...')
        interface.model = load_model('pure.h5')

    debug = True
    startDate = '2018/03/15' if debug else '2018/03/31'
    backDate = '2018/03/17' if debug else '2018/04/08'
    isStudent = True
    config.includeBack = True
    config.MAX_PASS = 100

    from _secret import secrets
    users = list(map(lambda x: packUserInfo(**x), secrets))
    userInfo = users[0]
    # Adjust here to fit your needs
    data = packInfo(toDate=startDate, toTime='20:30',
            backDate=backDate, backTime='20:30', 
            from_='台南', to_='左營', 
            tick_n=[0, 0, 0, 0, len(users)],
            isStudent=isStudent, incBack=config.includeBack)
    loop = True
    retry = 100 if loop else 1
    mBook = Book()

    def onfail():
        global retry
        print('OnFail with retry='+str(retry))
        time.sleep(0.4)
        retry -= 1
        if retry > 0:
            booking(**bParam)

    bParam = {
        'userInfo': userInfo,
        'data': data,
        'isStudent': isStudent,
        'onSuccess': mBook.add,
        'onFailure': onfail, 
    }

    booking(**bParam)

    if args.cancel:
        mBook.cancelAll()
    mBook.write()
