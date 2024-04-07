#include "crc.hpp"

#include <cstdint>
#include <iostream>
#include <iomanip>
using namespace std;
namespace crc16
{
    /**
     * @brief CRC16 Caculation function
     * @param[in] pchMessage : Data to Verify,
     * @param[in] dwLength : Stream length = Data + checksum
     * @return : CRC16 checksum
     */
    uint16_t Get_CRC16_Check_Sum(const uint8_t *pchMessage, uint32_t dwLength)
    {
        uint16_t crc_poly = 0xA001; // Bit sequence inversion of 0x8005
        uint16_t data_t = 0xFFFF;   // CRC register

        for (uint32_t i = 0; i < dwLength; i++)
        {
            data_t ^= pchMessage[i]; // 8-bit data

            for (uint8_t j = 0; j < 8; j++)
            {
                if (data_t & 0x0001)
                    data_t = (data_t >> 1) ^ crc_poly;
                else
                    data_t >>= 1;
            }
        }

        return data_t ^ 0xFFFF;
    }

    /**
     * @brief CRC16 Verify function
     * @param[in] pchMessage : Data to Verify,
     * @param[in] dwLength : Stream length = Data + checksum
     * @return : True or False (CRC Verify Result)
     */
    uint32_t Verify_CRC16_Check_Sum(const uint8_t *pchMessage, uint32_t dwLength)
    {
        uint16_t w_expected = 0;

        if ((pchMessage == nullptr) || (dwLength <= 2))
            return false;

        w_expected = Get_CRC16_Check_Sum(pchMessage, dwLength - 2);
        return (
            (w_expected & 0xff) == pchMessage[dwLength - 1] &&
            ((w_expected >> 8) & 0xff) == pchMessage[dwLength - 2]);
    }

    /**
     * @brief Append CRC16 value to the end of the buffer
     * @param[in] pchMessage : Data to Verify,
     * @param[in] dwLength : Stream length = Data + checksum
     * @return none
     */
    void Append_CRC16_Check_Sum(uint8_t *pchMessage, uint32_t dwLength)
    {
        uint16_t w_crc = 0;

        if ((pchMessage == nullptr) || (dwLength <= 2))
            return;
        // 发送数据包
        w_crc = Get_CRC16_Check_Sum(reinterpret_cast<uint8_t *>(pchMessage), dwLength - 2);

        pchMessage[dwLength - 1] = (uint8_t)(w_crc & 0x00ff);
        pchMessage[dwLength - 2] = (uint8_t)((w_crc >> 8) & 0x00ff);
    }

} // namespace crc16