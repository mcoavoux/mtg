#ifndef HASH_UTILS_H
#define HASH_UTILS_H


// 32 bit hashing (Jenkins)

#define __rot__(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

#define __mix__(a,b,c) \
{ \
a -= c;  a ^= __rot__(c, 4);  c += b; \
b -= a;  b ^= __rot__(a, 6);  a += c; \
c -= b;  c ^= __rot__(b, 8);  b += a; \
a -= c;  a ^= __rot__(c,16);  c += b; \
b -= a;  b ^= __rot__(a,19);  a += c; \
c -= b;  c ^= __rot__(b, 4);  b += a; \
}

#define __final__(a,b,c) \
{ \
c ^= b; c -= __rot__(b,14); \
a ^= c; a -= __rot__(c,11); \
b ^= a; b -= __rot__(a,25); \
c ^= b; c -= __rot__(b,16); \
a ^= c; a -= __rot__(c,4);  \
b ^= a; b -= __rot__(a,14); \
c ^= b; c -= __rot__(b,24); \
}

typedef unsigned int Int;

//Constant controlling Feature X vector size (Kernel Size)
//primes < 10^6
static const Int PRIME_MICRO = 499979;
static const Int PRIME_MINI = 999983;
//10^7 closest prime
static const Int PRIME_SMALL = 1999993;
//10^8 closest prime
static const Int PRIME_LARGE = 19999999;

//Max Table size (primes, use with modulus op)
static const Int KERNEL_SIZE  = PRIME_LARGE;


struct KernelIndexer{

    unsigned int operator()(Int a, Int b, Int c, Int d)const{

        __mix__(a,b,c);

        a += d;

        __final__(a,b,c);

        return c % KERNEL_SIZE;
    }
};



#endif // HASH_UTILS_H
