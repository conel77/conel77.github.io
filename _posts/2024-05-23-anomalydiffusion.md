---
layout: post
title: "Anomalydiffusion: Few-Shot Anomaly Image Generation with Diffusion Model (+Textual Inversion)"
image: framework_anodiff.jpg
date: 2024-05-23 14:27:00 +0900
tags: [anomaly image generation, diffusion, textual inversion]
categories: anomaly image generation
---


<p align="center">
  <img src="/images/framework_anodiff.png" alt="AnomalyDiffusion Framework" width="600"/>
</p>

textual inversion 에서 학습한 textual embedding 을 활용해서 anomaly 한 타입을 결정함.

근데 이와 같은 텍스트 인버전의 문제점은 특정지역으로만 위치한 anomaly 한 이미지를 만들어낸다는 것. 

→ 따라서 텍스트 임베딩을 두개의 부분으로 분해해서 하나는 spatial embedding 으로 어노멀리한 부분을 가리키는 anomaly mask로부터 인코딩을 하고, 다른 하나는 anomaly embedding 으로, 오로지 anomaly type 정보만을 인코딩을 하는 임베딩 두개로 나눈다. 

**Anomaly Embedding**

기존의 textual inversion 과 다른 점은, 전체 이미지를 보는 기존 모델과 다르게 여기서 사용하는 textual inversion 은 anomaly 한 지역만 보도록 한다. → masked textual inversion 

마스킹을 해서 어노멀리한 지역만을 모델이 볼 수 있도록 만들었다. 

**Spatial Embedding**

어노멀리한 지역의 정확한 정보를 제공하기 위해서 textual embedding 인데 마스크로부터 정확한 위치 정보를 받은 `e_s`를 사용한다. 

이미지를 일단 resnet50 에 넣어서 feature extract 를 한 후에 Feature pyramid network 를 사용해서 다양한 레이어의 피처들을 합쳤다. 마지막으로 FC layer 를 거친 피처를 text embedding 에다가 fuse

이 두 개를 concat 시켜서 text condition 으로 사용한다. 

근데 이런 방식을 사용했었을 때, 생성된 어노멀리 이미지들이 마스크와 제대로 정렬이 안될 때가 있다. → 이런 경우가 뒤의 downstream task 를 수행할 때 문제를 일으킬 수 있다.

이를 해결하기 위해 **attention re-weighting**을 적용한다.

디노이징 과정 동안 덜 주목받은 생성된 어노멀리에 대해 더 많은 attention 을 할당한다. 이 weight 를 부여하는 방법으로 adaptive attention weight map 을 사용한다. 

---

어노디프는 일단 textual inversion 방식으로 진행된다.

**textual inversion 이란?**

diffusion 학습 과정에서 text encoder 를 타고 diffusion model 에 컨디션을 줄 때, few shot dataset 의 패턴을 특정 텍스트 토큰에 학습시키기 위해서 unique identifier 를 업데이트 하는 과정에서 few shot 이미지와 생성한 이미지 간의 Loss 를 걸어 이 텍스트 토큰을 update 하는 방식이다.

→ 이렇게 되면 파인튜닝 방식으로 사용할 수 있으며, 해당 text encoder 부분만 업데이트 된다.

어노멀리 디퓨전에서는 총 2단계로 진행된다.

1. 첫번째 단계: 이미지 generation 학습  
   - anomaly embedding 과 spatial encoder 만 학습된다.
   - anomaly embedding 은 클래스와 타입별로 각각 학습되며, spatial encoder weight 는 1개만 사용해 학습된다.
   - 여기서 spatial encoder 는 단순 segmentation 네트워크로 사용되지 않고, anomaly mask 로부터 피처를 추출하여 anomaly embedding 과 동일 사이즈로 맞춘다.
   - 그러나 둘의 역할을 분리하는 로스가 없기 때문에 두 임베딩 간의 기능이 일부 섞일 가능성이 있다.

2. 두번째 단계: 마스크를 생성하는 학습  
   - 마스크 임베딩을 stage 1 과 동일한 방식으로 학습하는데, 클래스-타입별로 각각 spatial embedding 이 작동하는 부분에 들어간다.
   - 즉, 텍스쳐 인버전 방식으로 학습하여, 샘플링 시 해당 임베딩을 넣으면 마스크가 생성되도록 한다.

정리하면, **anomaly embedding 이 타입별로 각각 들어가고, spatial embedding 은 유니버셜하게 사용된다.**  
이상적으로 anomaly embedding 으로 타입별 특성이 구별되어야 하지만, 두 임베딩이 같이 학습되기 때문에 기능이 섞여서 다른 타입의 텍스쳐가 반영될 가능성이 있다.

---

**메모리 사용량 (Anodiff 예시):**

- **512x512 해상도**  
  - image train: 42GB (batch 4)  
  - mask train: 30GB (batch 4)  
  - image inference: 15GB  
  - mask inference: 14GB

- **256x256 해상도**  
  - image train: 18GB (batch 4)  
  - mask train: 10GB (batch 4)  
  - image inference: 10GB  
  - mask inference: 10GB

---

실행 예시:
CUDA_VISIBLE_DEVICES=0 python main.py --spatial_encoder_embedding --data_enhance --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml -t --actual_resume models/ldm/text2img-large/model.ckpt -n test --gpus 0 --init_word anomaly --mvtec_path=/home/MVTec