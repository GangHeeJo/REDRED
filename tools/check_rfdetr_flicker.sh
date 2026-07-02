#!/bin/bash
# 50ep+SAM2 RF-DETR 결과의 과다발화 9개 클래스 flicker 리플레이
# 실행: bash tools/check_rfdetr_flicker.sh
python tools/replay_event_detector.py --debug_log=output/debug_frame_counts_rfdetr.csv \
  pepperidge_farm_milano_cookies_double_chocolate \
  nature_valley_crunchy_oats_n_honey \
  nabisco_nilla_wafers \
  pepperidge_farm_milk_chocolate_macadamia_cookies \
  chewy_dips_peanut_butter \
  crayola_24_crayons \
  dove_white \
  a1_steak_sauce \
  aunt_jemima_original_syrup
