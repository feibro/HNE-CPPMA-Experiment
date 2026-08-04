#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <miracl/miracl.h>
/*extern "C" {
#include "miracl.h"
}
*/
namespace {

constexpr int N = 1000;
volatile unsigned int sink = 0;

using Bytes = std::vector<uint8_t>;

uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32U - n));
}

template <size_t L>
std::array<uint8_t, L> sha2(const uint8_t* data, size_t len, bool is224) {
    static constexpr uint32_t K[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
    };

    uint32_t h[8];
    if (is224) {
        uint32_t init[8] = {
            0xc1059ed8U, 0x367cd507U, 0x3070dd17U, 0xf70e5939U,
            0xffc00b31U, 0x68581511U, 0x64f98fa7U, 0xbefa4fa4U
        };
        std::memcpy(h, init, sizeof(h));
    } else {
        uint32_t init[8] = {
            0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
            0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
        };
        std::memcpy(h, init, sizeof(h));
    }

    Bytes msg(data, data + len);
    const uint64_t bit_len = static_cast<uint64_t>(len) * 8U;
    msg.push_back(0x80U);
    while ((msg.size() % 64U) != 56U) {
        msg.push_back(0U);
    }
    for (int i = 7; i >= 0; --i) {
        msg.push_back(static_cast<uint8_t>((bit_len >> (i * 8)) & 0xffU));
    }

    for (size_t off = 0; off < msg.size(); off += 64U) {
        uint32_t w[64]{};
        for (int i = 0; i < 16; ++i) {
            const size_t j = off + static_cast<size_t>(i) * 4U;
            w[i] = (static_cast<uint32_t>(msg[j]) << 24U) |
                   (static_cast<uint32_t>(msg[j + 1]) << 16U) |
                   (static_cast<uint32_t>(msg[j + 2]) << 8U) |
                   static_cast<uint32_t>(msg[j + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7U) ^ rotr(w[i - 15], 18U) ^ (w[i - 15] >> 3U);
            const uint32_t s1 = rotr(w[i - 2], 17U) ^ rotr(w[i - 2], 19U) ^ (w[i - 2] >> 10U);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6U) ^ rotr(e, 11U) ^ rotr(e, 25U);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t t1 = hh + s1 + ch + K[i] + w[i];
            const uint32_t s0 = rotr(a, 2U) ^ rotr(a, 13U) ^ rotr(a, 22U);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;

            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    std::array<uint8_t, L> out{};
    for (size_t i = 0; i < L; ++i) {
        out[i] = static_cast<uint8_t>((h[i / 4U] >> (24U - 8U * (i % 4U))) & 0xffU);
    }
    return out;
}

std::array<uint8_t, 28> sha224_local(const Bytes& msg) {
    return sha2<28>(msg.data(), msg.size(), true);
}

std::array<uint8_t, 32> sha256_miracl(const Bytes& msg) {
    ::sha256 sh{};
    shs256_init(&sh);
    for (uint8_t b : msg) {
        shs256_process(&sh, static_cast<int>(b));
    }

    char raw[32]{};
    shs256_hash(&sh, raw);

    std::array<uint8_t, 32> out{};
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<uint8_t>(raw[i]);
    }
    return out;
}

template <size_t L>
std::string hex_of(const std::array<uint8_t, L>& data) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string s;
    s.reserve(L * 2U);
    for (uint8_t b : data) {
        s.push_back(hex[b >> 4U]);
        s.push_back(hex[b & 0x0fU]);
    }
    return s;
}

void hash_self_test() {
    const std::string text = "abc";
    const Bytes msg(text.begin(), text.end());

    if (hex_of(sha224_local(msg)) != "23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7") {
        throw std::runtime_error("SHA-224 self-test failed");
    }
    if (hex_of(sha256_miracl(msg)) != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
        throw std::runtime_error("SHA-256 self-test failed");
    }
}

big bn(const char* hex) {
    big x = mirvar(0);
    std::vector<char> s(std::strlen(hex) + 1U);
    std::memcpy(s.data(), hex, s.size());
    if (cinstr(x, s.data()) == 0) {
        throw std::runtime_error("invalid hex integer");
    }
    return x;
}

struct Ec {
    const char* id;
    const char* name;
    const char* p;
    const char* a;
    const char* b;
    const char* gx;
    const char* gy;
    const char* k;
};

struct Op {
    std::string tag;
    std::string desc;
    double ms;
};

template <class F>
double bench(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        f(i);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double total = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return total / static_cast<double>(N);
}

epoint* base_point(big gx, big gy) {
    epoint* g = epoint_init();
    if (!epoint_set(gx, gy, 0, g)) {
        throw std::runtime_error("invalid base point");
    }
    return g;
}

std::vector<Op> run_ec(const Ec& ec) {
    big p = bn(ec.p);
    big a = bn(ec.a);
    big b = bn(ec.b);
    big gx = bn(ec.gx);
    big gy = bn(ec.gy);
    big k = bn(ec.k);
    big small = bn("10001");
    big two = bn("2");

    ecurve_init(a, b, p, MR_BEST);

    epoint* g = base_point(gx, gy);
    epoint* r = epoint_init();
    epoint* acc = epoint_init();
    ecurve_mult(two, g, acc);

    const std::string s = std::string("ECCexplement2 benchmark input for ") + ec.id + " on " + ec.name;
    Bytes msg(s.begin(), s.end());

    std::vector<Op> ops;
    ops.push_back({std::string("T_{ecm}^{") + ec.id + "}", "scalar multiplication",
                   bench([&](int) { ecurve_mult(k, g, r); })});

    ops.push_back({std::string("T_{eca}^{") + ec.id + "}", "point addition",
                   bench([&](int) { ecurve_add(g, acc); })});

    ops.push_back({std::string("T_{secm}^{") + ec.id + "}", "small scalar multiplication, k=0x10001",
                   bench([&](int) { ecurve_mult(small, g, r); })});

    if (std::strcmp(ec.id, "E2") == 0) {
        ops.push_back({"T_h^{E2}", "SHA-224",
                           bench([&](int i) {
                               msg[0] = static_cast<uint8_t>(i & 0xff);
                           const auto d = sha224_local(msg);
                           sink ^= d[static_cast<size_t>(i) % d.size()];
                       })});
    } else {
        ops.push_back({"T_h^{E3}", "SHA-256",
                       bench([&](int i) {
                           msg[0] = static_cast<uint8_t>(i & 0xff);
                           const auto d = sha256_miracl(msg);
                           sink ^= d[static_cast<size_t>(i) % d.size()];
                       })});
    }

    big x = mirvar(0);
    big y = mirvar(0);
    epoint_get(r, x, y);
    sink ^= static_cast<unsigned int>(size(x) ^ size(y));

    mirkill(x); mirkill(y);
    epoint_free(g); epoint_free(r); epoint_free(acc);
    mirkill(p); mirkill(a); mirkill(b); mirkill(gx); mirkill(gy);
    mirkill(k); mirkill(small); mirkill(two);
    return ops;
}

void print_ops(const Ec& ec, const std::vector<Op>& ops) {
    std::cout << ec.id << " (" << ec.name << ")\n";
    for (const auto& op : ops) {
        std::cout << "  " << std::left << std::setw(16) << op.tag
                  << "  " << std::setw(42) << op.desc
                  << "  avg = " << std::fixed << std::setprecision(6)
                  << op.ms << " ms\n";
    }
}

} // namespace

int main() {
    miracl* mip = mirsys(100, 0);
    if (mip == nullptr) {
        std::cerr << "MIRACL init failed\n";
        return 1;
    }
    mip->IOBASE = 16;
    hash_self_test();

    const Ec e2{
        "E2", "NIST P-256",
        "FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC",
        "5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B",
        "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
        "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
        "1F1E1D1C1B1A19181716151413121110FFEEDDCCBBAA99887766554433221100"
    };

    const Ec e3{
        "E3", "NIST P-224",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF000000000000000000000001",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFE",
        "B4050A850C04B3ABF54132565044B0B7D7BFD8BA270B39432355FFB4",
        "B70E0CBD6BB4BF7F321390B94A03C1D356C21122343280D6115C1D21",
        "BD376388B5F723FB4C22DFE6CD4375A05A07476444D5819985007E34",
        "1F1E1D1C1B1A19181716151413121110FFEEDDCCBBAA9988776655"
    };

    try {
        const auto e2_ops = run_ec(e2);
        const auto e3_ops = run_ec(e3);

        std::cout << "MIRACL ECC benchmark, each operation averaged over "
                  << N << " runs, unit: ms\n\n";
        print_ops(e2, e2_ops);
        std::cout << '\n';
        print_ops(e3, e3_ops);
        std::cout << "\nchecksum: " << sink << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        mirexit();
        return 1;
    }

    mirexit();
    return 0;
}
