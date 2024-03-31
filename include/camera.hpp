#include <MvCameraControl.h>
#include <MvSdkExport.h>
#include <opencv2/opencv.hpp>

namespace HIK
{
    class Camera
    {
        public:
            Camera();
            ~Camera();
            void open();
            cv::Mat capture();
        private:
            void* m_hDevHandle;
            MV_FRAME_OUT stOutFrame;
    };
} // namespace HIK
