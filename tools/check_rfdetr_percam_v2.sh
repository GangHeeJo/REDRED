#!/bin/bash
# skip=3/conf=0.5 기준으로 새로 미탐지된 5개 클래스 카메라별 원인 분석
# 실행: bash tools/check_rfdetr_percam_v2.sh
python tools/analyze_per_cam.py --per_cam output/per_cam_rfdetr.csv --focus \
  hunts_sauce \
  pepperidge_farm_milk_chocolate_macadamia_cookies \
  cheerios \
  chewy_dips_chocolate_chip \
  spam \
  a1_steak_sauce \
  aunt_jemima_original_syrup \
  dove_white
