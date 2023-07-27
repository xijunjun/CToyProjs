@echo off
set arg1=yolov8n.mnn
set arg2=test.jpg
echo Running example.exe with arguments %arg1% and %arg2%...
start "" "./yolov8_demo.exe" %arg1% %arg2%
echo Finished running example.exe.
pause



