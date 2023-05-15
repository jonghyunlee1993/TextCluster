# Text Analyzer: 주제 군집 분석 및 요약

- 엑셀 파일의 특정 컬럼을 읽어서 자동으로 군집 분석을 실시, 군집별 정보를 요약해주는 프로그램.
- 작동 원리
  - BERT 모델을 이용하여 텍스트의 의미 벡터 추출
  - 의미 벡터를 사전에 정의된 군집 수에 맞추어 K-means clustering 실시
  - 분류된 군집의 텍스트를 모아 chatGPT에 입력, 해당 군집의 주제를 분석
  - 분석된 결과에 따라 최종 엑셀 파일 저장

# 사용 방법

- 환경 구축

  - Python 터미널을 프로젝트 폴더로 이동한 후, 아래 커맨드 입력
    - `pip install -r requirements.txt`
  - `key/openai_key.txt` 에 자신의 openai api key 를 저장.
    - 방법은 [API 키 발급 방법]](https://www.daleseo.com/chatgpt-api-keys/ ) 참고

- 스크립트 실행

  - `python run_analyzer.py -project sample_data -coi 의견 -nc 6`
    - project: 특정 프로젝트의 이름. 예시의 입력 파일 이름은 sample_data.xlsx 이며, 최종 결과 값은 sample_data_resutls.xlsx 로 저장
    - coi: column of interest, 관심 컬럼의 이름. 분석하고자 하는 데이터가 들어있는 열의 이름. 예시에서는 의견 컬럼
    - nc: number of cluster, 분석하고자 하는 데이터의 군집 숫자 예상치. 결과를 보면서 유동적으로 늘리거나 줄이며 조정.


# 폴더 위계

![image-20230513152111841](/Users/jonghyunlee/Library/Application Support/typora-user-images/image-20230513152111841.png)

- 프로젝트 폴더
  - data
    - 분석하고자 하는 엑셀 파일
  - key
    - openai api 키를 저장하기 위한 텍스트 파일
  - results
    - 분석이 완료된 엑셀 파일



# 입력 파일 세팅

![image-20230513151557945](/Users/jonghyunlee/Library/Application Support/typora-user-images/image-20230513151557945.png)

- Sheet1에 A1 셀부터 데이터가 시작되도록 세팅
  - 공백이 있으면 안됨
  - 열 이름에는 띄어쓰기 없이 입력. 
    - 필요시 `핵심_가치` 처럼 _ 이용하여 입력 요망. (띄어쓰기를 제대로 인식하지 못할 가능성이 있음)

# 결과 해석

![image-20230513151339627](/Users/jonghyunlee/Library/Application Support/typora-user-images/image-20230513151339627.png)

- Summary 시트
  - 모든 결과가 종합적으로 정리되는 시트. 
    - Theme 밑에 있는 텍스트는 각 군집의 내용을 chatGPT를 이용하여 요약한 결과
    - 의견 밑에 있는 수치는 전체 중 해당 군집이 차지하는 비중 (%)
- Cluster_n
  - 각 군집으로 분류된 텍스트가 저장
  - 확인 후, 군집이 제대로 생성되지 않았다고 판단되면 군집의 수를 증가, 반대로 과도하게 나누어졌다고 판단되면 군집의 수를 줄임. 