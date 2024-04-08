#ifndef _PACKET_HPP_
#define _PACKET_HPP_

#include <cstdint>
#include <vector>

#include <crc.hpp>

#pragma pack(1)
struct SendPacket
{
    uint8_t header = 0xA5;
    bool tracking : 1;
    uint8_t id : 3;
    uint8_t armors_num : 3; // 2-balance 3-outpost 4-normal
    uint8_t reserved : 1;
    float yaw;
    float distance;
    uint8_t end = 0x0D;
    uint8_t newline = 0x0A;
    uint16_t crc_checksum = 0;
};

struct ReceivePacket
{
    uint8_t header = 0x5A;
    uint8_t detect_color : 1; // 0-red 1-blue
    bool reset_tracker : 1;
    uint8_t reserved : 6;
    uint16_t crc_checksum = 0;
};
#pragma pack()

void float2bytes(float f, uint8_t *byte)
{
    memcpy(byte, &f, sizeof(float));
}

float byte2float(const uint8_t *byte)
{
    float f;
    memcpy(&f, byte, sizeof(float));
    return f;
}

void sendPacket(Serial &serial, const SendPacket &packet)
{
    uint8_t buffer[sizeof(SendPacket)];
    buffer[0] = packet.header;
    buffer[1] = packet.tracking | (packet.id << 1) | (packet.armors_num << 4) | (packet.reserved << 7);
    float2bytes(packet.yaw, buffer + 2);
    float2bytes(packet.distance, buffer + 6);
    buffer[sizeof(SendPacket) - 4] = packet.end;
    buffer[sizeof(SendPacket) - 3] = packet.newline;
    buffer[sizeof(SendPacket) - 2] = packet.crc_checksum & 0xff;
    buffer[sizeof(SendPacket) - 1] = (packet.crc_checksum >> 8) & 0xff;
    // Debug用
    cout << hex << setw(2) << setfill('0') << (int)buffer[0] << " "
                << setw(2) << setfill('0') << (int)buffer[1] << " "
                << setw(2) << setfill('0') << (int)buffer[2] << " "
                << setw(2) << setfill('0') << (int)buffer[3] << " "
                << setw(2) << setfill('0') << (int)buffer[4] << " "
                << setw(2) << setfill('0') << (int)buffer[5] << " "
                << setw(2) << setfill('0') << (int)buffer[6] << " "
                << setw(2) << setfill('0') << (int)buffer[7] << " "
                << setw(2) << setfill('0') << (int)buffer[8] << " "
                << setw(2) << setfill('0') << (int)buffer[9] << " "
                << setw(2) << setfill('0') << (int)buffer[10] << " "
                << setw(2) << setfill('0') << (int)buffer[11] << " "
                << setw(2) << setfill('0') << (int)buffer[12] << " "
                << setw(2) << setfill('0') << (int)buffer[13] << " ->buffer" << endl;
    // 发送数据包
    serial.write(reinterpret_cast<const char *>(buffer), sizeof(SendPacket));
}

void receivePacket(Serial &serial, ReceivePacket &packet)
{
    uint8_t buffer[sizeof(ReceivePacket)];
    serial.read(reinterpret_cast<char *>(buffer), sizeof(ReceivePacket), 10);

    packet.header = buffer[0];
    packet.detect_color = buffer[1] >> 7;
    packet.reset_tracker = (buffer[1] >> 6) & 0x01;
    packet.reserved = buffer[1] & 0x3f;
    packet.crc_checksum = (0xffff & buffer[sizeof(ReceivePacket) - 1]) | (buffer[sizeof(ReceivePacket) - 2] << 8);
}

#endif