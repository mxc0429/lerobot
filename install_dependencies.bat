@echo off
echo ========================================
echo 安装所有依赖
echo ========================================
echo.

echo 正在安装基础依赖...
pip install numpy mujoco matplotlib

echo.
echo 正在安装机械臂依赖...
pip install feetech-servo-sdk pyserial

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.

echo 验证安装:
python -c "import numpy, mujoco, matplotlib; print('基础依赖 OK')"
python -c "import scservo_sdk, serial; print('机械臂依赖 OK')"

echo.
echo 所有依赖已安装完成！
echo 现在可以运行程序了。
echo.
pause
