## Run program
ROOT=$(pwd)
ROOT_img=$ROOT/"img_folder/sculpture"
ROOT_calib=$ROOT/"img_folder/sculpture/calibration.txt"
kp_type="SURF"
echo "Root of the img folder is set up as: "$ROOT_img &&
mkdir -p build && cd build && cmake .. && make clean &&
make -j4 && ./3DR --dir_img $ROOT_img --input_calib $ROOT_calib \
--keypoint_type $kp_type --launch_viewer=false