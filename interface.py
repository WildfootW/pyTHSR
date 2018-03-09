# coding: utf-8

import requests
import numpy as np
import pandas as pd
from PIL import Image
from lxml import etree

from core import WIDTH, HEIGHT
from image import Byte2Img
from handler import parse_output
from transform import preprocess
import config

'''Wrap Restful Api with functions'''

baseurl = 'https://irs.thsrc.com.tw'
lang = 'tw' # TODO: make it configurable
verify_error = {
    'en': u'Incorrect verification',
    'tw': u'檢測碼輸入錯誤',
    'jp': u'確認コードが誤っています',
}

def getSecurityURL(html):
    return html.xpath('//img[@id="BookingS1Form_homeCaptcha_passCode"]')[0].get('src')

def Image2Text(im):
    global model
    if isinstance(im, Image.Image):
        im = np.asarray(im)
    img = preprocess(im.reshape((1, HEIGHT, WIDTH, 1)))
    result = parse_output(model.predict(img))[0]
    print('Predict: ' + result)
    return result

def _bookMethod(rt):
    # 依時間
    return rt.xpath('//input[@id="bookingMethod1"]')[0].get('value')

def _seatPref(rt):
    # 靠窗優先
    return rt.xpath('//input[@id="seatRadio1"]')[0].get('value')

def getStatus(responseRt):
    '''check if the requested html doesn't have any error or not.
    
    Returns:
        True:  req success
        None:  req failed but Captcha match
        False: req failed, have a look at the error message ( through STDOUT )
    '''
    elems = responseRt.xpath('//div[@id="error"]/*/ul/li/span[@class="feedbackPanelERROR"]')
    if len(elems) == 0:
        return True
    
    print('\n'.join([ 'Error: ' + elem.text for elem in elems]))
    if not verify_error[lang] in ''.join([ elem.text for elem in elems]):
        return None
    return False

def booking(userInfo, data, isStudent=True, onSuccess=None, onFailure=None):
    if config.args.viz: import IPython
    query = '/IMINT/' + ('?student=university' if isStudent else '')
    try:
        with requests.Session() as sess:
            orderId = _doBooking(sess, userInfo, data, query)
    except AssertionError as e:
        print('EXCEPT: ' + str(e))
        onFailure is not None and onFailure()
    else:
        print("Mission complete")
        onSuccess is not None and onSuccess(orderId)

def cancel_book(_id, orderId):
    with requests.Session() as sess:
        _doCancel(sess, _id, orderId)

def _doBooking(sess, userInfo, data, query):
    res = sess.get(baseurl + query)

    # parse response HTML
    html = etree.HTML(res.text)
    BookingS1Form = html.xpath('//form[@id="BookingS1Form"]')[0]
    hidden1 = BookingS1Form.xpath('.//input[@type="hidden"]')[0].get('name')
    action = BookingS1Form.get('action')
    
    a_tag = html.xpath('//a[@id="BookingS1Form_homeCaptcha_reCodeLink"]')[0]
    href = a_tag.get('href').replace('ILinkListener', 'IBehaviorListener')
    post_href = href.split('=')[1]
    data['seatCon:seatRadioGroup'] = _seatPref(html)
    data['bookingMethod'] = _bookMethod(html)

    counter = 0
    canPass = False
    right_cap = False
    while not canPass:
        assert counter < config.MAX_PASS
        if not right_cap:
            sec_res = sess.get(baseurl + getSecurityURL(html))
            sec_image = Byte2Img(sec_res.content)
            if config.args.viz:  IPython.display.display(sec_image)

            # identify the securityCode
            securityCode = Image2Text(sec_image)
            data[hidden1] = ''
            data['homeCaptcha:securityCode'] = securityCode
        
        res_submit = sess.post(url=baseurl + action, data=data)
        res_root = etree.HTML(res_submit.text)
        canPass = getStatus(res_root)
    
        if canPass is None:
            right_cap = True
        elif not canPass:
            reCode = '&'.join(['wicket:interface:=' + post_href,])
            html = etree.HTML(sess.get(baseurl + '/IMINT/?' + reCode).text)
            counter += 1

    print('\nSecond Stage')
    BookingS2Form = res_root.xpath('//form[@id="BookingS2Form"]')[0]
    hidden2 = BookingS2Form.xpath('.//input[@type="hidden"]')[0].get('name')
    if config.args.viz:  
        # print table
        df = pd.read_html(etree.tostring(BookingS2Form.xpath('.//table')[0], method='HTML'), 
                          header=0, index_col=0)[0]
        IPython.display.display(df.fillna(''))

    # pick one value
    #row = BookingS2Form.xpath('//span[@id="BookingS2Form_TrainQueryDataViewPanel"]' + 
    #            '/table//tr[not(@class="section_subtitle")]')[0]
    train_radio = './/input[@name="TrainQueryDataViewPanel%s:TrainGroup"]'
    rname = BookingS2Form.xpath(train_radio % '')[0].get('value')
    rname2 = BookingS2Form.xpath(train_radio % '2')[0].get('value') if config.includeBack else ''
    
    action2 = BookingS2Form.get('action')
    res_submit2 = sess.post(url=baseurl + action2, data={
        hidden2: '',
        'TrainQueryDataViewPanel:TrainGroup': rname,
        'TrainQueryDataViewPanel2:TrainGroup': rname2,
        'SubmitButton': '確認車次'
    })
    res_root = etree.HTML(res_submit2.text)
    assert getStatus(res_root), 'Error Occur in stage2'

    print('\nFinal Stage')
    if config.args.viz:
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
    res_submit3 = sess.post(url=baseurl+action3, data=userInfo)
    res_root = etree.HTML(res_submit3.text)
    assert getStatus(res_root), 'Error Occur in stage3'
    orderid = res_root.xpath('//span/table[@class="table_details"]//td[@class="content_key"]/span')[0].text
    return orderid

def _doCancel(sess, _id, orderId):
    page_query = '/IMINT/?wicket:bookmarkablePage=:tw.com.mitac.webapp.thsr.viewer.History'
    # enter history page
    res = sess.get(baseurl + page_query)
    res_rt = etree.HTML(res.text)
    assert getStatus(res_rt), "Fail"
    radio_id = res_rt.xpath('//input[@id="idInputRadio1"]')[0].get('value')
    action = res_rt.xpath('//form[@id="HistoryForm"]')[0].get('action')
    
    # send cancel requests
    res_submit = sess.post(url=baseurl+action, data={
            'SubmitButton': '登入查詢',
            'idInputRadio': radio_id,
            'idInputRadio:rocId': _id,
            'orderId': orderId,
        })
    res_root = etree.HTML(res_submit.text)
    assert getStatus(res_root), ""
    print('NextPage')
    
    action = res_root.xpath('//form[@id="HistoryDetailsForm"]')[0].get('action')
    res = sess.post(url=baseurl+action, data={
            'TicketProcessButtonPanel:CancelSeatsButton': '取消訂位', })
    res_root = etree.HTML(res.text)
    assert getStatus(res_root), "Fail -2"
    
    action = res_root.xpath('//form[@id="HistoryDetailsCancelForm"]')[0].get('action')
    res = sess.post(url=baseurl+action, data={
            'agree': 'on',
            'SubmitButton': '下一步',
        })
    res_root = etree.HTML(res.text)
    assert getStatus(res_root), "Fail last"
    title = res_root.xpath('//td[@class="payment_title"]/span')[0].text
    print('取消成功')
    print('Title: ' + unicode(title))

