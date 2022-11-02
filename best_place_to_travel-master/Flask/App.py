#-*-coding: utf-8-*-
from unittest import result
from flask import Flask, render_template, request
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image
import cv2

app=Flask(__name__)



classes = ['0.index',
           '01.숭례문',
           '02.경주_불국사_다보탑',
           '03.경주_태종무열왕릉비',
           '04.경주_첨성대',
           '05.청자_상감운학문_매병',
           '06.경주_불국사_금동비로자나불좌상',
           '07.금동미륵보살반가사유상',
           '08.천마총_금관',
            '09.창경궁_자격루',
           '10.백제_금동대향로',
           '11.선덕대왕신종',
           '12.익산_미륵사지_석탑',
           '13.훈민정음',
           '14.청주_용두사지_철당간',
           '15.안동_하회탈_및_병산탈'
          ]
infomation = {
    '01.숭례문' : '숭례문은 조선시대 도성을 둘러싸고 있던 성곽의 정문으로, 일명 남대문(南大門)이라고도 하는데, 서울 도성의 사대문 가운데 남쪽에 있기 때문에 붙여진 이름이다. 1962년 12월 20일에 국보로 지정되었고, 문화재청 숭례문 관리소에서 관리하고 있다.',
    '02.경주_불국사_다보탑' : '경상북도 경주시 진현동(進峴洞) 불국사 경내에 있는 남북국시대의 화강암제 석탑. 535년(법흥왕 22년)에 불국사가 창건한 후 751년(경덕왕 10년) 김대성의 발원으로 불국사가 중건될 때 옆에 있는 불국사 3층 석탑과 함께 수축(修築)한 것으로 추정된다.',
    '03.경주_태종무열왕릉비' : '경주 태종무열왕릉비는 경상북도 경주시 서악동에 위치한 무열왕릉 앞에 있는, 화강암으로 만들어진 비석이다. 661년에 신라의 태종 무열왕이 승하하고 문무왕이 바로 뒤이어 즉위했는데, 이 해에 무열왕릉비가 세워졌다고 한다',
    '04.경주_첨성대' : '경상북도 경주시 인왕동에 있는 선덕여왕때 지어진 신라시대의 천문대. 신라 왕궁 터인 반월성의 북서쪽 성곽에서 약 300 m 떨어진 지점에 있다. 국보 제31호이고, 그 원형을 유지하는 것 가운데 현존하는 세계에서 가장 오래된 천문대이다.',
    '05.청자_상감운학문_매병' : '고려 매병(梅甁)은 중국 송(宋)나라 매병에서 유래된 것이지만, 12세기경에 이르러서는 고려만의 풍만하면서도 유연한 선의 아름다움이 나타난다. 이러한 고려 매병의 양식은 이 작품에서 세련미의 극치를 보여주고 있다.',
    '06.경주_불국사_금동비로자나불좌상' : '경주시 토함산 기슭에 자리잡은 불국사는 통일신라 경덕왕 10년(751) 김대성의 발원에 의해 창건된 사찰로, 『삼국유사』에 의하면 김대성은 현세의 부모를 위해 불국사를, 전생의 부모를 위해 석굴암 석굴을 창건하였다고 한다. 불국사 비로전에 모셔져 있는 높이 1.77m의 이 불상은 진리의 세계를 두루 통솔한다는 의미를 지닌 비로자나불을 형상화한 것이다.',
    '07.금동미륵보살반가사유상' : '국립중앙박물관에 있는 금동미륵보살반가사유상(국보)과 함께 국내에서는 가장 큰 금동반가사유상으로 높이가 93.5㎝이다. 1920년대에 경주에서 발견되었다고 전하나 근거가 없으며, 머리에 3면이 둥근 산 모양의 관(冠)을 쓰고 있어서 ‘삼산반가사유상(三山半跏思惟像)’으로도 불린다.',
    '08.천마총_금관' : '천마총에서 발견된 신라 때 금관이다. 천마총은 경주 고분 제155호 무덤으로 불리던 것을 1973년 발굴을 통해 금관, 팔찌 등 많은 유물과 함께 천마도가 발견되어 천마총이라 부르게 되었다.',
    '09.창경궁_자격루' : '자격루(自擊漏)는 조선 세종 때의 물시계로, 자동으로 시간마다 종이 울리도록 한 국가 표준시계이다. 장영실과 김조 등이 2년 간 제작하여 세종 16년 (1434년) 8월 5일 (음력 7월 1일) 완성·발표하였다. 이후 중종 31년에 이전의 자격루를 개량하여 다시 제작하였으며, 이들은 경복궁과 창덕궁의 보루각에서 보관되었다.',
    '10.백제_금동대향로' : '백제금동대향로는 1993년 12월 12일(일) 충청남도 부여군 부여읍 능산리에서 주차장 공사를 하던 중 발견된 백제의 향로이다. 이후 조사 결과 해당 향로가 발견된 주차장 공사 현장이 백제 시대 왕실의 사찰이 있었던 곳으로 밝혀졌다. 대향로를 언제 제작했는지 정확히는 알 수 없지만 대략 6세기 말-7세기 초라고 추정한다.',
    '11.선덕대왕신종' : '성덕대왕신종(聖德大王神鍾)은 신라 시대에 만들어진 범종이다. 설화에 따라 에밀레종으로 부르거나 봉덕사(奉德寺)에 걸려 있던 종이라 하여 봉덕사종이라 부르기도 한다. 1962년 12월 20일 대한민국의 국보 제29호로 지정되었다.',
    '12.익산_미륵사지_석탑' : '익산 미륵사지 석탑(益山 彌勒寺址 石塔)은 전라북도 익산시 금마면 미륵사지에 있으며, 한국에 남아있는 석탑 중 가장 오래된 석탑으로 국보 제11호로 지정되어 있다. 제 무왕의 재위기간 중인 639년에 만들어진 이 석탑은 백제 석탑의 시원 형식(始原形式)이라고 불리며, 여러 면에서 한국 석탑 전체의 출발점이라 할 수 있다.',
    '13.훈민정음' : '세종대왕의 훈민정음을 우리말로 번역한 책. 현재 1459년 만들어진 원본이 서강대학교 도서관에, 그 필사본들이 서울대학교 도서관, 고려대학교 도서관, 세종 대왕 기념 사업회, 일본 궁내성 등에 소장되어 있다. 또 다른 명칭으로는 훈민정음주해본(訓民正音註解本)이라고도 부른다.',
    '14.청주_용두사지_철당간' : '청주 용두사지 철당간이라는 명칭으로 국보 제41호로 지정되어 있으며, 국보 제106호 계유명전씨아미타불비상, 국보 제297호 안심사 영산회 괘불탱와 함께 청주시의 3개 뿐인 국보이다.',
    '15.안동_하회탈_및_병산탈' : '경상북도 안동군 하회마을과 그 이웃인 병산마을에 전해 내려오는 탈로서 현존하는 가장 오래된 탈놀이 가면이다. 하회탈로는 11개가 전해지는데 주지 2개, 각시, 중, 양반, 선비, 초랭이, 이매, 부네, 백정, 할미 탈이 있다. 병산탈은 총각, 별채, 떡다리 탈이 있었다고 하나 분실되어 현재 2개가 남아 있다.'
}

address={
    '01.숭례문' : '서울특별시 중구 세종대로 40',
    '02.경주_불국사_다보탑' : '경상북도 경주시 불국로 385',
    '03.경주_태종무열왕릉비' : '경주시 대경로 4859',
    '04.경주_첨성대' : '경상북도 경주시 인왕동 839-1',
    '05.청자_상감운학문_매병' : '서울 성북구 성북로 102-11',
    '06.경주_불국사_금동비로자나불좌상' : '경상북도 경주시 불국로 385',
    '07.금동미륵보살반가사유상' : '서울특별시 용산구 서빙고로 137',
    '08.천마총_금관' : '경북 경주시 일정로 186',
    '09.창경궁_자격루' : '서울특별시 중구 세종대로 99',
    '10.백제_금동대향로' : '충남 부여군 부여읍 금성로 5',
    '11.선덕대왕신종' : '경북 경주시 일정로 186',
    '12.익산_미륵사지_석탑' : '전북 익산시 금마면 기양리 97번지',
    '13.훈민정음' : '서울특별시 성북구 성북로 102-11',
    '14.청주_용두사지_철당간' : '충청북도 청주시 상당구 남문로2가 48-19',
    '15.안동_하회탈_및_병산탈' : '경상북도 안동시 민속촌길 13'
}

recommend={
    '01.숭례문' : 'https://www.tripadvisor.co.kr/RestaurantsNear-g294197-d3808315-Sungnyemun_Gate-Seoul.html',
    '02.경주_불국사_다보탑' : 'https://www.tripadvisor.co.kr/RestaurantsNear-g297888-d320364-Bulguksa_Temple-Gyeongju_Gyeongsangbuk_do.html',
    '03.경주_태종무열왕릉비' : 'https://www.tripadvisor.co.kr/RestaurantsNear-g297888-d2471041-Royal_Tomb_of_King_Taejong_Muyeol-Gyeongju_Gyeongsangbuk_do.html',
    '04.경주_첨성대' : 'https://www.tripadvisor.co.kr/RestaurantsNear-g297888-d2476679-Cheomseongdae_Observatory-Gyeongju_Gyeongsangbuk_do.html',
    '05.청자_상감운학문_매병' : 'https://m.joseilbo.com/news/view_2020.htm?newsid=380945',
    '06.경주_불국사_금동비로자나불좌상' : 'https://www.tripadvisor.co.kr/RestaurantsNear-g297888-d320364-Bulguksa_Temple-Gyeongju_Gyeongsangbuk_do.html',
    '07.금동미륵보살반가사유상' : 'https://www.tripadvisor.co.kr/RestaurantsNear-g294197-d325043-National_Museum_of_Korea-Seoul.html',
    '08.천마총_금관' : 'https://www.tripadvisor.co.kr/RestaurantsNear-g297888-d2093002-Gyeongju_National_Museum-Gyeongju_Gyeongsangbuk_do.html',
    '09.창경궁_자격루' : 'https://www.mangoplate.com/top_lists/1345_changgyeonggung',
    '10.백제_금동대향로' : 'https://www.tripadvisor.co.kr/RestaurantsNear-g946498-d2476641-Buyeo_National_Museum-Buyeo_gun_Chungcheongnam_do.html',
    '11.선덕대왕신종' : 'https://www.tripadvisor.co.kr/RestaurantsNear-g297888-d2093002-Gyeongju_National_Museum-Gyeongju_Gyeongsangbuk_do.html',
    '12.익산_미륵사지_석탑' : 'https://www.tripadvisor.co.kr/RestaurantsNear-g2024820-d2478945-Iksan_Mireuksa_Temple_Site-Iksan_Jeollabuk_do.html',
    '13.훈민정음' : 'https://m.joseilbo.com/news/view_2020.htm?newsid=380945',
    '14.청주_용두사지_철당간' : 'https://blog.naver.com/ep1971kk/222607182733',
    '15.안동_하회탈_및_병산탈' : 'https://www.tripadvisor.co.kr/RestaurantsNear-g1074321-d3805477-oa30-Andong_Folk_Museum-Andong_Gyeongsangbuk_do.html'
}

accommodation={
    '01.숭례문' : 'https://www.tripadvisor.co.kr/HotelsNear-g294197-d3808315-Sungnyemun_Gate-Seoul.html',
    '02.경주_불국사_다보탑' : 'https://www.booking.com/landmark/kr/bulguk-temple.ko.html',
    '03.경주_태종무열왕릉비' : 'https://www.booking.com/landmark/kr/royal-tomb-of-king-taejong-muyeol.ko.html',
    '04.경주_첨성대' : 'https://www.booking.com/landmark/kr/cheomseongdae.ko.html',
    '05.청자_상감운학문_매병' : 'https://www.tripadvisor.co.kr/HotelsNear-g294197-d3805580-Gansong_Art_Museum-Seoul.html',
    '06.경주_불국사_금동비로자나불좌상' : 'https://www.booking.com/landmark/kr/bulguk-temple.ko.html',
    '07.금동미륵보살반가사유상' : 'https://www.booking.com/landmark/kr/national-museum-of-korea.ko.html',
    '08.천마총_금관' : 'https://kr.trip.com/hotels/gyeongju-gyeongju-national-museum/hotels-c3675m6740957/',
    '09.창경궁_자격루' : 'https://www.booking.com/landmark/kr/changgyeonggung.ko.html',
    '10.백제_금동대향로' : 'https://kr.trip.com/hot/%EA%B5%AD%EB%A6%BD+%EB%B6%80%EC%97%AC+%EB%B0%95%EB%AC%BC%EA%B4%80+%EA%B7%BC%EC%B2%98+%ED%98%B8%ED%85%94/',
    '11.선덕대왕신종' : 'https://www.booking.com/landmark/kr/gyeongju-national-museum1.ko.html',
    '12.익산_미륵사지_석탑' : 'https://kr.trip.com/hot/%EC%9D%B5%EC%82%B0%EB%AF%B8%EB%A5%B5%EC%82%AC%EC%A7%80%EC%84%9D%ED%83%91+%EA%B7%BC%EC%B2%98+%ED%98%B8%ED%85%94/',
    '13.훈민정음' : 'https://www.tripadvisor.co.kr/HotelsNear-g294197-d3805580-Gansong_Art_Museum-Seoul.html',
    '14.청주_용두사지_철당간' : 'https://www.booking.com/placestostay/city/kr/chongju.ko.html',
    '15.안동_하회탈_및_병산탈' : 'https://kr.trip.com/hot/%EC%95%88%EB%8F%99%EB%AF%BC%EC%86%8D%EB%B0%95%EB%AC%BC%EA%B4%80+%EA%B7%BC%EC%B2%98+%ED%98%B8%ED%85%94/'
}

@app.route('/')
def base():
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def upload():
    f_image= request.files['file']
    fname = f_image.filename[:-4]
    img = Image.open(f_image)
    img = img.resize((224,224))
    img.save('./static/upload/'+fname+'.jpg')
    img = np.array(Image.open('./static/upload/'+fname+'.jpg'))/255.0
    img_resized = cv2.resize(img, (224,224))
    model = keras.models.load_model('./static/best_weight75.h5')
    result = classes[model.predict(img_resized.reshape(-1,224,224,3)).argmax() + 1]
    info=infomation[result]
    juso=address[result]
    res=recommend[result]
    hotel=accommodation[result]
    mtime = int(os.stat('./static/upload/'+fname+'.jpg').st_mtime)
    return render_template('result1.html', result = result, juso=juso, info=info, res=res, hotel=hotel, fname= fname, mtime = mtime)
if __name__=='__main__':
    app.run(debug=True)
