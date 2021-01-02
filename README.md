# 국회 입법활동 (National_assembly)
## 개요
* [dacon](https://dacon.io/competitions/official/235679/data/) 의 종료된 데이터를 이용해 국회활동을 알아본다.

## requirements
* pip install -r requirements.txt
* KoNLPy - [설치방법](https://konlpy.org/ko/latest/install/)

## path
* analysis : User\data

## Report

### history

#### 대수별 국회의원수
* 대수별 국회의원수는 아래와 같이 증가함을 보인다.
* ![이미지](https://github.com/kdj6394/National_assembly/blob/main/img/history/%EB%8C%80%EC%88%98%EB%B3%84_%EA%B5%AD%ED%9A%8C%EC%9D%98%EC%9B%90%EC%88%98.png?raw=true)

#### 출신 지역별 국회의원수
* 출신 지역은 아래와 같이 경북 > 경남 > 전남 > 서울 > 경기... 순으로 나타난다.
* ![이미지](https://github.com/kdj6394/National_assembly/blob/main/img/history/%EC%B6%9C%EC%8B%A0%EC%A7%80%EC%97%AD.png?raw=true)

#### 대수별 발의법의안 건수
* 대수별 발의법안건수는 아래와 같이 16대 이후 급격한 증가를 보인다.
* 특히 21대 국회는 현재 진행중이므로 건수가 낮음이 보인다.
*![이미지](https://github.com/kdj6394/National_assembly/blob/main/img/history/%EB%8C%80%EC%88%98%EB%B3%84%EB%B0%9C%EC%9D%98%EB%B2%95%EC%9D%98%EC%95%88%EA%B1%B4%EC%88%98.png?raw=true)

#### 급격히 증가한 16대 이후의 월별 발의법의안 건수
* 월별 발의법의안 건수는 6월 이후 최대치를 보여주는것으로 나타나며 9월경에도 소폭 증가함을 보인다.
* ![이미지](https://github.com/kdj6394/National_assembly/blob/main/img/history/16%EB%8C%80%EC%9D%B4%ED%9B%84%EC%9B%94%EB%B3%84%EB%B0%9C%EC%9D%98%EB%B2%95%EC%9D%98%EC%95%88%EA%B1%B4%EC%88%98.png?raw=true)

#### 대수별 발의법의안 처리상태
* 8대 : 박정희 정권, 따라서 8대 국회는 해산되고 대신해서 비상국무회의에서 대부분의 안건을 처리했다.
* 대수가 지남에 따라 폐기,철회,부결의 비율이 상승 가결률은 점점 하락
* 1[이미지](https://github.com/kdj6394/National_assembly/blob/main/img/history/%EB%8C%80%EC%88%98%EB%B3%84%EB%B0%9C%EC%9D%98%EB%B2%95%EC%9D%98%EC%95%88%EC%B2%98%EB%A6%AC%EB%B0%A9%EB%B2%95%EB%B9%84%EC%9C%A8.png?raw=true)

#### 발의법의안 Top5
* 아래와같이 5명이 등장한다.
* 여기서 이 5명이 top5로 나온 것은 연임을 했기 때문입니다.
* 이명수 : 18,19,20,21대 국회의원 당선
* 황주홍 : 19,20대 국회의원 당선
* 강창일 : 17,18,19,20대 국회의원 당선
* 오제세 : 17,18,19,20대 국회의원 당선
* 김우남 : 17,18,19대 국회의원 당선
* ![이미지](https://github.com/kdj6394/National_assembly/blob/main/img/history/%EB%B0%9C%EC%9D%98%EB%B2%95%EC%9D%98%EC%95%88Top5.png?raw=true)

#### 최근 5개 대수별 법률안 종류
* 도로교통법, 근로기준법, 감염병관련법이 최근 5개 대수에서 발의안건수가 가장 많은듯하다.
* 가장 많은 발의안건을 하는 법은 조세특례제한법, 공직선거법, 국회법이 대표적이다.
* 도로교통법은 18대에서 20대까지 5위에서 3위를 차지하면서 안건수가 상승하였다.
* 근로기준법 또한 20대에서 21대로 2단계 상승하면서 순위가 올랐다.
* 최근 코로나로 21대 국회에서는 기존에 top5에 없었던 감염병관련 법안이 4위로 올라왔음을 볼 수 있다.
* ![이미지](https://github.com/kdj6394/National_assembly/blob/main/img/history/%EC%A7%80%EB%82%9C5%EB%8C%80%EC%88%98%EB%B2%95%EC%9D%98%EB%B0%9C%EC%9D%98%EC%95%88%EC%A2%85%EB%A5%98.png?raw=true)

#### 대수별 본회의 처리안건수
* 아래처럼 대수가 늘어남에 따라 처리안건수또한 증가함을 보인다.
* ![이미지](https://github.com/kdj6394/National_assembly/blob/main/img/history/%EB%8C%80%EC%88%98%EB%B3%84%EB%B3%B8%ED%9A%8C%EC%9D%98%EC%B2%98%EB%A6%AC%EC%95%88%EA%B1%B4%EC%88%98.png?raw=true)

#### 대수별 총득표수 찬성,반대,기원의 비율
| age | mean_yes_ratio | mean_no_ratio | mean_blank_ratio |
|---|---|---|---|
|0     |1        |0.000000       |0.000000          |0.000000|
|1     |2        |0.000000       |0.000000          |0.000000|
|2     |3        |0.000000       |0.000000          |0.000000|
|3     |4        |0.000000       |0.000000          |0.000000|
|4     |5        |0.000000       |0.000000          |0.000000|
|5     |6        |0.000000       |0.000000          |0.000000|
|6     |7        |0.000000       |0.000000          |0.000000|
|7     |8        |0.000000       |0.000000          |0.000000|
|8     |9        |0.000000       |0.000000          |0.000000|
|9    |10        |0.000000       |0.000000          |0.000000|
|10   |11        |0.000000       |0.000000          |0.000000|
|11   |12        |0.000000       |0.000000          |0.000000|
|12   |13        |0.000000       |0.000000          |0.000000|
|13   |14        |0.000000       |0.000000          |0.000000|
|14   |15        |0.000000       |0.000000          |0.000000|
|15   |16        |0.000000       |0.000000          |0.000000|
|16   |17        |0.000000       |0.000000          |0.000000|
|17   |18        |0.000000       |0.000000          |0.000000|
|18   |19        |0.000000       |0.000000          |0.000000|
|19   |20        |0.175875       |0.001942          |0.004914|
|20   |21        |0.617761       |0.005027          |0.021356|

* 대수별 찬성,반대, 기권수 비율은 20대, 21대 외에는 자료가 없기때문에 20대와 21대의 평균을 구하면 아래와 같다.

| |age | kind | mean_ratio|
|---|---|---|---|
|0  | 20 |   yes   |     0.18|
|1  | 20 |    no   |     0.00|
|2  | 20 | blank   |     0.00|
|3  | 21 |   yes   |     0.62|
|4  | 21 |    no   |     0.01|
|5  | 21 | blank   |     0.02|

* 20대보다 3배 많은 21대에서 평균적으로 60%찬성률을 보이지만 결측치가 많아 지표로 쓰긴 어렵다.
* ![이미지](https://github.com/kdj6394/National_assembly/blob/main/img/history/20,21%EB%8C%80%ED%88%AC%ED%91%9C%EB%B9%84%EC%9C%A8%ED%8F%89%EA%B7%A0.png?raw=true)

#### 대수별 안건이 제안후 공포까지의 시간
* 모든 대수의 평균 소요일은 약 80일이나 최근 4개,17-20대수는 평균적으로 제안에서 공포되기까지 걸리는 시간이 150일 즉, 4-5개월이 걸린다는 것을 알 수 있다.
* ![이미지](https://github.com/kdj6394/National_assembly/blob/main/img/history/%EB%8C%80%EC%88%98%EB%B3%84%EC%A0%9C%EC%95%88%EC%97%90%EC%84%9C%EA%B3%B5%ED%8F%AC%EA%B9%8C%EC%A7%80%EC%8B%9C%EA%B0%84.png?raw=true)

#### 제안일과 공포일의 월별,요일별 분포
* 제안일 기준
* 연말로 갈수록 제안건수가 많아지며 11월 12월이 최대임을 알 수있다.
* 요일별 제안건수는 평일 월~금 모두 비슷하고 주말은 확연히 적다.
* ![이미지](https://github.com/kdj6394/National_assembly/blob/main/img/history/%EC%9B%94%EB%B3%84,%EC%9A%94%EC%9D%BC%EB%B3%84%EC%A0%9C%EC%95%88%EC%9D%BC%EC%88%98.png?raw=true)

* 공포일 기준
* 12월달에 공포를 제일 많이 하고 연초에 조금 많다는 것을 볼 수 있다.
* 요일별로는 화요일이 많이 공포되는 요일이다
* ![이미지](https://github.com/kdj6394/National_assembly/blob/main/img/history/%EC%9B%94%EB%B3%84,%EC%9A%94%EC%9D%BC%EB%B3%84%EA%B3%B5%ED%8F%AC%EC%9D%BC%EC%88%98.png?raw=true)

### COVID-19
#### 최근 3대 국회 총 발의법의안 수
* 21대 국회 총 발의법의안 수:  4015 건
* 20대 국회 총 발의법의안 수:  21594 건
* 19대 국회 총 발의법의안 수:  15444 건

#### 최근 3대 국회 감염병 관련 발의법의안 Top5
* 코로나19 관련 법에는 '감염병의 예방 및 관리에 관한 법률 일부개정법률안', '의료법 일부개정법률안' 이 있다.

|최근 3대 국회감염병 발의법안수|---|
|---|---|
|제 21대 감염병 관련 발의법의안 TOP 5|---|
|감염병의 예방 및 관리에 관한 법률 일부개정법률안      |               55|
|의료법 일부개정법률안                               |      36|
|의료급여법 일부개정법률안                            |        3|
|공중보건 위기대응 의료제품의 개발촉진 및 긴급대응을 위한 의료제품 공급 특례법안  |   2|
|보건의료기본법 일부개정법률안                        |          2|
|---|---|
|제 20대 감염병 관련 발의법의안 TOP 5|---|
|의료법 일부개정법률안                 |            183|
|감염병의 예방 및 관리에 관한 법률 일부개정법률안      |        54|
|의료급여법 일부개정법률안                    |        20|
|의료사고 피해구제 및 의료분쟁 조정 등에 관한 법률 일부개정법률안   |  14|
|의료기사 등에 관한 법률 일부개정법률안           |         14|
|---|---|
|제 19대 감염병 관련 발의법의안 TOP 5|---|
|의료법 일부개정법률안                |    113|
|감염병의 예방 및 관리에 관한 법률 일부개정법률안    | 51|
|의료급여법 일부개정법률안              |     15|
|의료기사 등에 관한 법률 일부개정법률안      |     12|
|보건의료기본법 일부개정법률안            |     11|

* ![이미지](https://github.com/kdj6394/National_assembly/blob/main/img/covid-19/%EC%B5%9C%EA%B7%BC3%EB%8C%80%EA%B5%AD%ED%9A%8C%EA%B0%90%EC%97%BC%EB%B3%91%EA%B4%80%EB%A0%A8%EB%B0%9C%EC%9D%98%EB%B2%95%EC%9D%98%EC%95%88Top5.png?raw=true)

#### 최근 3대 국회 감염병관련 입법 비율
* 특히 21대의 입법비율이 19대와 20대의 2배이다.
* 21대 국회의 코로나 3법 입법 비율:  2.267 %
* 20대 국회의 코로나 3법 입법 비율:  1.125 %
* 19대 국회의 코로나 3법 입법 비율:  1.114 %

* ![이미지](https://github.com/kdj6394/National_assembly/blob/main/img/covid-19/%EC%B5%9C%EA%B7%BC3%EB%8C%80%EA%B5%AD%ED%9A%8C%EA%B0%90%EC%97%BC%EB%B3%91%EA%B4%80%EB%A0%A8%EB%B2%95%EC%95%88%EC%9E%85%EB%B2%95%EB%B9%84%EC%9C%A8.png?raw=true)