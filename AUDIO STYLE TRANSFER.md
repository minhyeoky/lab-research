[AUDIO STYLE TRANSFER](https://arxiv.org/pdf/1710.11385.pdf)
ICASSP 2018

리뷰

## 초록

vision은 style transfer가 활발하다. 

audio 에 적용.

1. 텍스쳐 모델: 오디오의 스타일을 나타내는 통계를 추출하기 위한 sound texture model
2. 스타일을 합치기 위한 synthetic model 

두 가지를 제안.

## 서론, 관련연구

기존의 전통적인 방법 ~ Neural network

1. stylized 라기 보단 두 소리가 그냥 섞인 것처럼 들림.
2. prosodic speech 에 적용하는 시도. 낮은 레벨의 texture는 입힐 수 있었지만, 억양이나 감정등은 담기 어려웠다.
3. reorchestraion - symbolic music (digital 음악이 아니고 노트로 표현된 음악들)에 대한 것이라 도메인이 다르다.

 Gatys의 [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)의 frame work를 차용해서

 최적화 단계 + 단순화 된 loss(style에 대해서만) 를 사용함

 스타일의 통계와 로스를 정의하기 위해서 여러 CNN 아키텍쳐를 조사함(vision에서 영감을 받아 오디오 분야에 적용된 것들) 

## 2. 문제정의 및 토의

 **A Neural Algorithm of Artistic Style**

 Gatys가 제안한 ST는 content image와 style이미지 각각에서 feature map들을 추출하고 저장한다. 추출한 각각의 feature를 사용하여 새로운 이미지를 만드는 것이 주요 문제이다. (content fature + relevant statistic of style features)
 => 적합한 two-fold loss? 반복적인 최소화로 만들어진 노이즈 이미지로 시작한다. -> 결론? 적으로 최종이미지는 content image의 전체적인 구조를 가지면서 style이미지의 texture와 color를 보존한다.

 **오디오와 관련된 스타일 트랜스퍼 선행 연구**

 비젼과 달리 wide, shallow random 한 네트워크 사용. (4096 필터를 가진 하나의 레이어)

1. 1D 의 오디오를 STFT를 사용하여 2D로 변환
2. 그 결과의 spectrogram 표현은 phase information을 제외하고 생각한다면 2D 이미지로 생각이 가능함
3. 벡터로 구성된 1D signal(frequeny)로 취급할 수도 있어
   4 = 이미지의 컬러채널은 audio분야에서는 frequency channel로 대체가 가능하다.
4. 랜덤하게 초기화한 spectrogram을 content 와 style audio에서 추출한 feature들을 사용한 loss function으로 최적화함
5. **이 연구의 결과의 한계점은, content 와 style의 보존여부가 명확하지 않다는 점**

 **Style과 content란?**

 명확하게 정의된 것은 없다.

- task와 context에 dependent 하다.
- vision에서 적당히 정의하자면, space-invariant intra-patch statics 라고 함 ***intra-patch는 무슨 용어인지 모르겠음***

 오디오에서는 더 애매한데, 단어, context, speaker 에 개성등 영향을 많이 받음

 음악이면은 그래도 content: 노래, style: 악기 또는 장르 등으로 볼 수 있다.

## 3. 제안한 프레임워크

 ![image](https://user-images.githubusercontent.com/12870549/62996797-ece53580-bea0-11e9-816c-9573ff1d5081.png)

###  

 모델은 크게

1. Style extraction
2. Style transfer

 로 구성

1. STFT등의 기법을 통해 전처리
2. sound texture 모델을 통해 texture statics 를 추출(NN또는 그냥 엔지니어링)
3. synthesize - by optimization algorithm
4. post processing. -> play!



**기존**: 랜덤 노이즈를 content + style loss로 최적화

**이 논문**: original audio를 사용. Initial audio는 content이고 최적화하는 것은 style loss뿐



이 논문에서는 content loss는 GD가 잘 일어나지 않게 되는 방해요소로써 필요치 않아서 삭제 했고, 이런 구조로써 content를 더 잘 보존할 수 있게 되었다.



#### NN based approach

x: input audio signal (raw or spectrogram)

$F_l = [f_{l,k}]^{K_l}_{k=1}$: 레이어 l에서 $K_l$ activation map 행렬

$G_l = F_l^TF_l$

$L(x;x_{style}) = \sum_{l\in L}||G_l(x) - G_l(x_{style})||^2_F$: content + style의 각각의 레이어에서의 G의 L2 norm loss function

위 로스함수를 GD를 통해 최적화.

**모델 테스트**

1. VGG: raw -> spectrogram -> x3 (채널을 만들기 위해) -> ?

2. SoundNet: SOTA모델, raw데이터 사용

3. **Wide-Shallow-Random**: 최근연구에서 학습되지 않은 shallow NN이 style statics를 위해 사용 될 수 잇다고 함, one-layer CNN with 4096 filters. 

   **input:**2D spectrogram

#### Auditory 

전통적 방법. 해석을 위해 사용.

pass

## 4. 실혐결과

![image-20190814152239711](http://ww3.sinaimg.cn/large/006tNc79gy1g5z6r4dx14j30dg0db11w.jpg)

1. original content로 초기화하는게 되게 좋았다.
2. Spectrogram 으로 분석
3. VGG는 별로
4. Soundnet: 나쁘지 않았지만 스타일이 별로 안 묻어나고 노이즈도 있었다.
5. shallow net with random filter가 오히려 나았음
6. McDermott's model(?) 이 젤 좋음
7. 고양이 소리가 spectrogram에 그대로 보임
8. 등 case stuies…..

