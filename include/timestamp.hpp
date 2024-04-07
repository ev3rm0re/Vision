#ifndef _TIMESTAMP_HPP_
#define _TIMESTAMP_HPP_

typedef long long timestamp_t;

class Timestamp {
    public:
        static timestamp_t now();
};

#endif