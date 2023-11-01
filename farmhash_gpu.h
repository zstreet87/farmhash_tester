// Copyright (c) 2014 Google, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// FarmHash, by Geoff Pike

#ifndef FARM_HASH_GPU_H_
#define FARM_HASH_GPU_H_

#include <cstdint>
#include <string.h>   // for memcpy and memset
#include <cassert>

#define NAMESPACE_FOR_HASH_FUNCTIONS_GPU util_gpu
#define DEVICE_MODIFIER __device__ __host__

#define uint32_in_expected_order(x) (x)
#define uint64_in_expected_order(x) (x)

#define STATIC_INLINE DEVICE_MODIFIER inline

namespace NAMESPACE_FOR_HASH_FUNCTIONS_GPU {

STATIC_INLINE uint64_t Fetch64(const char *p) {
  uint64_t result;
  memcpy(&result, p, sizeof(result));
  return uint64_in_expected_order(result);
}

STATIC_INLINE uint32_t Fetch32(const char *p) {
  uint32_t result;
  memcpy(&result, p, sizeof(result));
  return uint32_in_expected_order(result);
}

STATIC_INLINE uint64_t BasicRotate64(uint64_t val, int shift) {
  // Avoid shifting by 64: doing so yields an undefined result.
  return shift == 0 ? val : ((val >> shift) | (val << (64 - shift)));
}

STATIC_INLINE uint64_t Rotate64(uint64_t val, int shift) {
  return BasicRotate64(val, shift);
}

}  // namespace NAMESPACE_FOR_HASH_FUNCTIONS_GPU

// FARMHASH PORTABILITY LAYER: debug mode or max speed?
// One may use -DFARMHASH_DEBUG=1 or -DFARMHASH_DEBUG=0 to force the issue.

struct Pair {
  uint64_t first;
  uint64_t second;
};

namespace NAMESPACE_FOR_HASH_FUNCTIONS_GPU {

// Some primes between 2^63 and 2^64 for various uses.
static const uint64_t k0 = 0xc3a5c85c97cb3127ULL;
static const uint64_t k1 = 0xb492b66fbe98f273ULL;
static const uint64_t k2 = 0x9ae16a3b2f90404fULL;

}  // namespace NAMESPACE_FOR_HASH_FUNCTIONS_GPU

using namespace NAMESPACE_FOR_HASH_FUNCTIONS_GPU;
namespace farmhashna_gpu {
#undef Fetch
#define Fetch Fetch64

#undef Rotate
#define Rotate Rotate64

STATIC_INLINE uint64_t ShiftMix(uint64_t val) {
  return val ^ (val >> 47);
}

STATIC_INLINE uint64_t HashLen16(uint64_t u, uint64_t v, uint64_t mul) {
  // Murmur-inspired hashing.
  uint64_t a = (u ^ v) * mul;
  a ^= (a >> 47);
  uint64_t b = (v ^ a) * mul;
  b ^= (b >> 47);
  b *= mul;
  return b;
}

STATIC_INLINE uint64_t HashLen0to16(const char *s, size_t len) {
  if (len >= 8) {
    uint64_t mul = k2 + len * 2;
    uint64_t a = Fetch(s) + k2;
    uint64_t b = Fetch(s + len - 8);
    uint64_t c = Rotate(b, 37) * mul + a;
    uint64_t d = (Rotate(a, 25) + b) * mul;
    return HashLen16(c, d, mul);
  }
  if (len >= 4) {
    uint64_t mul = k2 + len * 2;
    uint64_t a = Fetch32(s);
    return HashLen16(len + (a << 3), Fetch32(s + len - 4), mul);
  }
  if (len > 0) {
    uint8_t a = s[0];
    uint8_t b = s[len >> 1];
    uint8_t c = s[len - 1];
    uint32_t y = static_cast<uint32_t>(a) + (static_cast<uint32_t>(b) << 8);
    uint32_t z = len + (static_cast<uint32_t>(c) << 2);
    return ShiftMix(y * k2 ^ z * k0) * k2;
  }
  return k2;
}

// This probably works well for 16-byte strings as well, but it may be overkill
// in that case.
STATIC_INLINE uint64_t HashLen17to32(const char *s, size_t len) {
  uint64_t mul = k2 + len * 2;
  uint64_t a = Fetch(s) * k1;
  uint64_t b = Fetch(s + 8);
  uint64_t c = Fetch(s + len - 8) * mul;
  uint64_t d = Fetch(s + len - 16) * k2;
  return HashLen16(Rotate(a + b, 43) + Rotate(c, 30) + d,
                   a + Rotate(b + k2, 18) + c, mul);
}

// Return a 16-byte hash for 48 bytes.  Quick and dirty.
// Callers do best to use "random-looking" values for a and b.
STATIC_INLINE Pair WeakHashLen32WithSeeds(
    uint64_t w, uint64_t x, uint64_t y, uint64_t z, uint64_t a, uint64_t b) {
  a += w;
  b = Rotate(b + a + z, 21);
  uint64_t c = a;
  a += x;
  a += y;
  b += Rotate(a, 44);
  return Pair{a + z, b + c};
}

// Return a 16-byte hash for s[0] ... s[31], a, and b.  Quick and dirty.
STATIC_INLINE Pair WeakHashLen32WithSeeds(
    const char* s, uint64_t a, uint64_t b) {
  return WeakHashLen32WithSeeds(Fetch(s),
                                Fetch(s + 8),
                                Fetch(s + 16),
                                Fetch(s + 24),
                                a,
                                b);
}

// Return an 8-byte hash for 33 to 64 bytes.
STATIC_INLINE uint64_t HashLen33to64(const char *s, size_t len) {
  uint64_t mul = k2 + len * 2;
  uint64_t a = Fetch(s) * k2;
  uint64_t b = Fetch(s + 8);
  uint64_t c = Fetch(s + len - 8) * mul;
  uint64_t d = Fetch(s + len - 16) * k2;
  uint64_t y = Rotate(a + b, 43) + Rotate(c, 30) + d;
  uint64_t z = HashLen16(y, a + Rotate(b + k2, 18) + c, mul);
  uint64_t e = Fetch(s + 16) * mul;
  uint64_t f = Fetch(s + 24);
  uint64_t g = (y + Fetch(s + len - 32)) * mul;
  uint64_t h = (z + Fetch(s + len - 24)) * mul;
  return HashLen16(Rotate(e + f, 43) + Rotate(g, 30) + h,
                   e + Rotate(f + a, 18) + g, mul);
}

DEVICE_MODIFIER uint64_t Hash64(const char *s, size_t len) {
  const uint64_t seed = 81;
  if (len <= 32) {
    if (len <= 16) {
      return HashLen0to16(s, len);
    } else {
      return HashLen17to32(s, len);
    }
  } else if (len <= 64) {
    return HashLen33to64(s, len);
  }

  // For strings over 64 bytes we loop.  Internal state consists of
  // 56 bytes: v, w, x, y, and z.
  uint64_t x = seed;
  uint64_t y = seed * k1 + 113;
  uint64_t z = ShiftMix(y * k2 + 113) * k2;
  Pair v = {0, 0};
  Pair w = {0, 0};
  x = x * k2 + Fetch(s);

  // Set end so that after the loop we have 1 to 64 bytes left to process.
  const char* end = s + ((len - 1) / 64) * 64;
  const char* last64 = end + ((len - 1) & 63) - 63;
  assert(s + len - 64 == last64);
  do {
    x = Rotate(x + y + v.first + Fetch(s + 8), 37) * k1;
    y = Rotate(y + v.second + Fetch(s + 48), 42) * k1;
    x ^= w.second;
    y += v.first + Fetch(s + 40);
    z = Rotate(z + w.first, 33) * k1;
    v = WeakHashLen32WithSeeds(s, v.second * k1, x + w.first);
    w = WeakHashLen32WithSeeds(s + 32, z + w.second, y + Fetch(s + 16));
    auto tmp = z;
    z = x;
    x = tmp;
    s += 64;
  } while (s != end);
  uint64_t mul = k1 + ((z & 0xff) << 1);
  // Make s point to the last 64 bytes of input.
  s = last64;
  w.first += ((len - 1) & 63);
  v.first += w.first;
  w.first += v.first;
  x = Rotate(x + y + v.first + Fetch(s + 8), 37) * mul;
  y = Rotate(y + v.second + Fetch(s + 48), 42) * mul;
  x ^= w.second * 9;
  y += v.first * 9 + Fetch(s + 40);
  z = Rotate(z + w.first, 33) * mul;
  v = WeakHashLen32WithSeeds(s, v.second * mul, x + w.first);
  w = WeakHashLen32WithSeeds(s + 32, z + w.second, y + Fetch(s + 16));
  auto tmp = z;
  z = x;
  x = tmp;
  return HashLen16(HashLen16(v.first, w.first, mul) + ShiftMix(y) * k0 + z,
                   HashLen16(v.second, w.second, mul) + x,
                   mul);
}

}  // namespace farmhashna_gpu
namespace NAMESPACE_FOR_HASH_FUNCTIONS_GPU {

// Fingerprint function for a byte array.
DEVICE_MODIFIER uint64_t Fingerprint64(const char* s, size_t len) {
  return farmhashna_gpu::Hash64(s, len);
}

}  // namespace NAMESPACE_FOR_HASH_FUNCTIONS_GPU

#endif // FARM_HASH_GPU_H_
