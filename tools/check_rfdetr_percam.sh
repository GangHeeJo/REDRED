#!/bin/bash
# RF-DETR per_cam_log 기반 quorum/whitelist 재산출 대상 클래스 분석
# 실행: bash tools/check_rfdetr_percam.sh
python tools/analyze_per_cam.py --per_cam output/per_cam_rfdetr.csv --focus \
  dove_white \
  campbells_chicken_noodle_soup \
  nature_valley_crunchy_oats_n_honey \
  nabisco_nilla_wafers \
  pepperidge_farm_milk_chocolate_macadamia_cookies \
  chewy_dips_peanut_butter \
  crayola_24_crayons \
  a1_steak_sauce \
  aunt_jemima_original_syrup \
  pepperidge_farm_milano_cookies_double_chocolate
