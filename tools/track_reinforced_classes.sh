#!/bin/bash
# 이번 재학습(2차, SAM2 3단계문턱)에서 집중 보강한 클래스들의 에폭별 AP/F1 추이 확인
# WEAK_CLASS_IDS: campbells/cheerios/hunts_sauce/chewy_dips_chocolate_chip/aunt_jemima
# TIMING_CLASS_IDS: bumblebee/haribo/milano/chewy_dips_peanut_butter/lindt
LOG=$(ls -t train_rfdetr_*.log | head -1)
echo "로그 파일: $LOG"
echo ""
grep -nE "campbells_chicken_noodle_soup|cheerios|hunts_sauce|chewy_dips_chocolate_chip|aunt_jemima_original_syrup|bumblebee_albacore|haribo_gold_bears_gummi_candy|pepperidge_farm_milano_cookies_double_chocolate|chewy_dips_peanut_butter|lindt_excellence_cocoa_dark_chocolate" "$LOG"
