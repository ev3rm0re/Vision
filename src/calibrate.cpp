#include <vector>
#include <opencv2/opencv.hpp>
#include <camera.hpp>

using namespace std;
using namespace cv;

void calibrate() {
    // 棋盘格参数
    Size boardSize(8, 5);        // 内部角点数（对应8x5个棋盘格）
    float square_size = 0.027f;
    
    // 3D坐标生成（单位：米）
    vector<Point3f> obj;
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            obj.emplace_back(j * square_size, i * square_size, 0);  // [1,3](@ref)
        }
    }

    // 图像采集
    vector<vector<Point3f>> objectPoints;
    vector<vector<Point2f>> imagePoints;
    HIK::Camera camera;
    camera.open();
    
    Mat frame;
    int count = 0;
    while (true) {
        camera.cap(&frame);
        if (frame.empty()) break;

        // 角点检测
        vector<Point2f> corners;
        bool found = findChessboardCorners(frame, boardSize, corners);
        if (found) {
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            // 亚像素优化（参数优化）[1](@ref)
            TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001);
            cornerSubPix(gray, corners, Size(11,11), Size(-1,-1), criteria);
            
            objectPoints.push_back(obj);
            imagePoints.push_back(corners);
            
            // 可视化
            drawChessboardCorners(frame, boardSize, corners, found);
            putText(frame, "Captured: " + to_string(objectPoints.size()), 
                    Point(20,40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
        }

        imshow("Calibration", frame);
        char key = waitKey(30);
        if (key == 27 || objectPoints.size() >= 30) break;  // 至少采集30张[1](@ref)
    }
    camera.close();

    // 标定验证
    if (objectPoints.size() < 6) {
        cerr << "Insufficient calibration images (min 6)" << endl;
        return;
    }

    // 执行标定（添加标志位优化）[1](@ref)
    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;
    int flags = CALIB_FIX_ASPECT_RATIO | CALIB_ZERO_TANGENT_DIST;
    double rms = calibrateCamera(objectPoints, imagePoints, frame.size(), 
                                cameraMatrix, distCoeffs, rvecs, tvecs, flags);
    
    // 保存参数[1,3](@ref)
    FileStorage fs("calibration.yaml", FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix << "distortion_coefficients" << distCoeffs;
    fs.release();

    // 重投影误差计算[1,2](@ref)
    double total_error = 0;
    for (size_t i=0; i<objectPoints.size(); ++i) {
        vector<Point2f> projected_points;
        projectPoints(objectPoints[i], rvecs[i], tvecs[i], 
                      cameraMatrix, distCoeffs, projected_points);
        double error = norm(imagePoints[i], projected_points, NORM_L2);
        total_error += error*error;
    }
    cout << "Reprojection error: " << sqrt(total_error/objectPoints.size()) 
         << " pixels" << endl;

    // 畸变校正测试[1,6](@ref)
    Mat undistorted;
    undistort(frame, undistorted, cameraMatrix, distCoeffs);
    imshow("Original vs Corrected", undistorted);
    waitKey(0);
}