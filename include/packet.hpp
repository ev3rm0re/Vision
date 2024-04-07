#ifndef PACKET_HPP
#define PACKET_HPP

#include <cstdint>
#include <vector>

#pragma pack(1)
struct SendPacket
{
    uint8_t header = 0x5A;
    float yaw;
    float pitch;
    float distance;
    uint8_t tail = 0x5B;
};

struct ReceivePacket
{
    uint8_t header = 0x5A;
    float yaw;
    float pitch;
    float distance;
    uint8_t tail = 0x5B;
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

void sendPacket(Serial& serial, const SendPacket& packet) {
    uint8_t buffer[sizeof(SendPacket)];
    buffer[0] = packet.header;
    float2bytes(packet.yaw, buffer + 1);
    float2bytes(packet.pitch, buffer + 5);
    float2bytes(packet.distance, buffer + 9);
    buffer[sizeof(SendPacket) - 1] = packet.tail;
    
    serial.write(reinterpret_cast<const char*>(buffer), sizeof(SendPacket));
}

void receivePacket(Serial& serial, ReceivePacket& packet) {
    uint8_t buffer[sizeof(ReceivePacket)];
    serial.read(reinterpret_cast<char*>(buffer), sizeof(ReceivePacket), 10);
    
    packet.header = buffer[0];
    packet.yaw = byte2float(buffer + 1);
    packet.pitch = byte2float(buffer + 5);
    packet.distance = byte2float(buffer + 9);
    packet.tail = buffer[sizeof(ReceivePacket) - 1];
}

#endif