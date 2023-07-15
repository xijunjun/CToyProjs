adb push flycv_test /data/local/tmp/
adb push ./run.sh /data/local/tmp/
adb shell  chmod 777 /data/local/tmp/flycv_test
adb shell  chmod 777 /data/local/tmp/run.sh
adb shell sh  /data/local/tmp/run.sh
echo batfinish
pause