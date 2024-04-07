#ifndef _CRC_HPP_
#define _CRC_HPP_

#include <cstdint>

namespace crc16
{
    uint16_t Get_CRC16_Check_Sum(const uint8_t *pchMessage, uint32_t dwLength);
    /**
     * @brief CRC16 Verify function
     * @param[in] pchMessage : Data to Verify,
     * @param[in] dwLength : Stream length = Data + checksum
     * @return : True or False (CRC Verify Result)
     */
    uint32_t Verify_CRC16_Check_Sum(const uint8_t *pchMessage, uint32_t dwLength);

    /**
     * @brief Append CRC16 value to the end of the buffer
     * @param[in] pchMessage : Data to Verify,
     * @param[in] dwLength : Stream length = Data + checksum
     * @return none
     */
    void Append_CRC16_Check_Sum(uint8_t *pchMessage, uint32_t dwLength);

} // namespace crc16

#endif // _CRC_HPP_