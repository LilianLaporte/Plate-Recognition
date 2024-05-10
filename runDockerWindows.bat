@ECHO off
ECHO "%cd%"

REM Initialise variables
SET HOSTPATH=%cd%
SET CONTAINERPATH=/home/imageprocessingcourse

for /f "delims=[] tokens=2" %%a in ('ping -4 -n 1 %ComputerName% ^| findstr [') do set NetworkIP=%%a


docker build -t cse2225image:latest .



REM Remove older container
docker rm cse2225container



REM Run new one
docker run --name cse2225container -it -p 1984:1984 -v "%HOSTPATH%":"%CONTAINERPATH%":rw -e DISPLAY=%NetworkIP%:0  cse2225image:latest

