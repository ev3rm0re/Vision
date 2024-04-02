#include "camera.hpp"

namespace HIK
{
    Camera::Camera()
    {
        isCameraOpened = false;
        handle = NULL;
    }

    Camera::~Camera()
    {
        if (isCameraOpened)
        {
            close();
        }
        if (handle != NULL)
        {
            MV_CC_DestroyHandle(handle);
        }
    }

    bool Camera::open()
    {
        MV_CC_DEVICE_INFO_LIST stDeviceList;
        memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_EnumDevices fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }
        if (stDeviceList.nDeviceNum > 0)
        {
            for (unsigned int i = 0; i < stDeviceList.nDeviceNum; i++)
            {
                std::cout << "Device: " << i << std::endl;
                MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
                if (NULL == pDeviceInfo)
                {
                    break;
                }
                std::cout << "Device: " << i << std::endl;
            }
        }
        else
        {
            std::cerr << "No device found!" << std::endl;
            return false;
        }

        // 选择相机
        unsigned int nIndex = 0;
        nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[nIndex]);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_CreateHandle fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 打开相机
        nRet = MV_CC_OpenDevice(handle);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_OpenDevice fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 关闭触发模式
        nRet = MV_CC_SetEnumValue(handle, "TriggerMode", MV_TRIGGER_MODE_OFF);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_SetEnumValue fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 设置图像格式
        nRet = MV_CC_SetEnumValue(handle, "PixelFormat", PixelType_Gvsp_BGR8_Packed);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_SetEnumValue fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 获取图像大小
        memset(&stParam, 0, sizeof(MVCC_INTVALUE));
        nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_GetIntValue fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        }
        nPayloadSize = stParam.nCurValue;

        // 初始化图像信息
        MV_FRAME_OUT_INFO_EX stImageInfo = {0};
        memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));
        pData = (unsigned char*)malloc(nPayloadSize);
        if (NULL == pData)
        {
            std::cerr << "malloc fail!" << std::endl;
        }
        memset(pData, 0, sizeof(pData));

        // 开始取流
        nRet = MV_CC_StartGrabbing(handle);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_StartGrabbing fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }
        return true;
    }

    cv::Mat Camera::capture()
    {
        nPayloadSize = stParam.nCurValue;
        memset(pData, 0, sizeof(pData));
        nRet = MV_CC_GetOneFrameTimeout(handle, pData, nPayloadSize, &stImageInfo, 1000);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_GetOneFrameTimeout fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return cv::Mat();
        }
        if (stImageInfo.enPixelType == PixelType_Gvsp_BGR8_Packed)
        {
            cv::Mat frame(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, pData);
            return frame;
        }
        else if (stImageInfo.enPixelType == PixelType_Gvsp_Mono8)
        {
            cv::Mat frame(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC1, pData);
            return frame;
        }
        else
        {
            std::cerr << "Unsupported pixel type!" << std::endl;
            return cv::Mat();
        }
    }

    void Camera::close()
    {
        MV_CC_StopGrabbing(handle);
        MV_CC_CloseDevice(handle);
    }
}