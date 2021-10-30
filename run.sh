cd build && make && cd ..
echo "==============make 结束，开始运行==============="
./build/ACMP /home/zhujun/MVS/data/scannet/scans_test/scene0707_00
# ./build/ACMP living_room
# ./build/ACMP /home/zhujun/MVS/data/scannet/scans_test/scene0707_00
# ./build/ACMP /media/zhujun/share/MVS/scene0707-00
./build/gauss_filter 