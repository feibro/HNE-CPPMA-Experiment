#include <NTL/ZZ.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Mat = std::vector<long>;   // 行优先存储的矩阵
using Vec = std::vector<long>;   // 模 q 向量
using VecZ = std::vector<long>;  // 有符号小整数向量，用于高斯采样和范数检查

volatile std::uint64_t sink = 0; // 防止编译器优化掉被测操作

struct Opt {
    long N = 256;          // 一般格参数 N
    long M = 17408;        // 一般格参数 M
    long K = 128;          // 一般格参数 K
    long q = 12289;        // 模数 q
    int iters = 1000;      // 每个操作测试次数
    long samp_dim = 17408; // T_sp/T_norm 默认按 M 维向量测试
    double sigma = 3.2;    // 离散高斯采样标准差
    long dg_tail = 0;      // 离散高斯采样截断边界；0 表示自动选择
    long B = 4096;         // 范数检查阈值
    std::uint64_t seed = 20260627;
    std::string csv;
    std::string only;      // 只测试某一个操作；为空时按 TABLE I 顺序全部测试
};

struct Result {
    std::string name;
    double avg_ms;
};

[[noreturn]] void usage(const char* exe) {
    std::cerr
        << "用法: " << exe << " [--N 256] [--M 17408] [--K 128] [--q 12289]\n"
        << "          [--iters 1000] [--samp-dim 17408] [--sigma 3.2] [--dg-tail auto] [--B 4096]\n"
        << "          [--seed 20260627] [--csv results.csv] [--only operation] [--quick]\n\n"
        << "说明:\n"
        << "  默认参数对应论文表格中的 (N,M,K,q)=(256,17408,128,12289)，每项测试 1000 次。\n"
        << "  --quick 仅用于快速检查编译和输出格式，会改用小参数和较少轮数。\n";
    std::exit(1);
}

long parse_long(const char* s, const std::string& key) {
    try {
        std::size_t pos = 0;
        long v = std::stol(s, &pos);
        if (pos != std::strlen(s)) throw std::invalid_argument("bad");
        return v;
    } catch (...) {
        throw std::runtime_error("参数 " + key + " 的值不是合法整数: " + s);
    }
}

double parse_double(const char* s, const std::string& key) {
    try {
        std::size_t pos = 0;
        double v = std::stod(s, &pos);
        if (pos != std::strlen(s)) throw std::invalid_argument("bad");
        return v;
    } catch (...) {
        throw std::runtime_error("参数 " + key + " 的值不是合法浮点数: " + s);
    }
}

Opt parse(int argc, char** argv) {
    Opt o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const std::string&) -> const char* {
            if (i + 1 >= argc) usage(argv[0]);
            ++i;
            return argv[i];
        };

        if (a == "--N") o.N = parse_long(need(a), a);
        else if (a == "--M") o.M = parse_long(need(a), a);
        else if (a == "--K") o.K = parse_long(need(a), a);
        else if (a == "--q") o.q = parse_long(need(a), a);
        else if (a == "--iters") o.iters = static_cast<int>(parse_long(need(a), a));
        else if (a == "--samp-dim") o.samp_dim = parse_long(need(a), a);
        else if (a == "--sigma") o.sigma = parse_double(need(a), a);
        else if (a == "--dg-tail") o.dg_tail = parse_long(need(a), a);
        else if (a == "--B") o.B = parse_long(need(a), a);
        else if (a == "--seed") o.seed = static_cast<std::uint64_t>(parse_long(need(a), a));
        else if (a == "--csv") o.csv = need(a);
        else if (a == "--only") o.only = need(a);
        else if (a == "--quick") {
            // 快速自检参数：只用于确认环境、编译和输出顺序，不能作为论文数据。
            o.N = 32;
            o.M = 512;
            o.K = 16;
            o.q = 12289;
            o.iters = 10;
            o.samp_dim = o.M;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
        } else {
            throw std::runtime_error("未知参数: " + a);
        }
    }

    if (o.N <= 0 || o.M <= 0 || o.K <= 0 || o.q <= 1 || o.iters <= 0 || o.samp_dim <= 0 || o.sigma <= 0.0) {
        throw std::runtime_error("N/M/K/q/iters/samp-dim/sigma 必须为正数，且 q>1。");
    }
    if (o.dg_tail < 0) throw std::runtime_error("dg-tail 不能为负数。");
    return o;
}

std::size_t checked_size(long r, long c, const char* what) {
    const auto rr = static_cast<unsigned long long>(r);
    const auto cc = static_cast<unsigned long long>(c);
    if (rr != 0 && cc > std::numeric_limits<std::size_t>::max() / rr) {
        throw std::runtime_error(std::string("维度过大，无法分配: ") + what);
    }
    return static_cast<std::size_t>(rr * cc);
}

inline std::size_t idx(long cols, long i, long j) {
    return static_cast<std::size_t>(i) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(j);
}

inline long red(long long x, long q) {
    long r = static_cast<long>(x % q);
    return r < 0 ? r + q : r;
}

inline long add_mod(long a, long b, long q) {
    return NTL::AddMod(a, b, q);
}

inline long mul_mod(long a, long b, long q) {
    return NTL::MulMod(a, b, q);
}

void fill_mod(std::vector<long>& a, long q, std::mt19937_64& rng) {
    std::uniform_int_distribution<long> dist(0, q - 1);
    for (auto& x : a) x = dist(rng);
}

std::uint64_t pick(const std::vector<long>& a) {
    if (a.empty()) return 0;
    const std::size_t p = (a.size() * 1315423911ULL + sink) % a.size();
    return static_cast<std::uint64_t>(a[p]);
}

std::uint64_t pick_z(const std::vector<long>& a) {
    if (a.empty()) return 0;
    const std::size_t p = (a.size() * 2654435761ULL + sink) % a.size();
    return static_cast<std::uint64_t>(static_cast<long long>(a[p]) + 0x9e3779b97f4a7c15ULL);
}

struct DGSampler {
    long tail = 0;
    long double total = 0.0L;
    std::vector<long> vals;
    std::vector<long double> cdf;

    explicit DGSampler(double sigma, long user_tail) {
        // 严格离散高斯采样：直接在整数支撑上按 exp(-x^2/(2 sigma^2)) 构造 CDT。
        // user_tail=0 时采用约 128-bit 安全尾界，截断后的尾部概率可忽略。
        const long double sig = static_cast<long double>(sigma);
        if (user_tail > 0) {
            tail = user_tail;
        } else {
            const long double sec = 128.0L;
            tail = static_cast<long>(std::ceil(sig * std::sqrt(2.0L * sec * std::log(2.0L))));
            tail = std::max<long>(tail, 16);
        }

        vals.reserve(static_cast<std::size_t>(2 * tail + 1));
        cdf.reserve(static_cast<std::size_t>(2 * tail + 1));
        const long double den = 2.0L * sig * sig;
        for (long x = -tail; x <= tail; ++x) {
            const long double w = std::exp(-(static_cast<long double>(x) * x) / den);
            total += w;
            vals.push_back(x);
            cdf.push_back(total);
        }
    }

    long sample(std::mt19937_64& rng) const {
        std::uniform_real_distribution<long double> unif(0.0L, total);
        const long double u = unif(rng);
        const auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        const std::size_t pos = static_cast<std::size_t>(std::distance(cdf.begin(), it));
        return vals[std::min(pos, vals.size() - 1)];
    }
};

// ------------------------- 基本线性代数操作 -------------------------

void mv_mul(const Mat& A, const Vec& x, Vec& y, long rows, long cols, long q) {
    for (long i = 0; i < rows; ++i) {
        long long acc = 0;
        const auto base = static_cast<std::size_t>(i) * static_cast<std::size_t>(cols);
        for (long j = 0; j < cols; ++j) {
            acc += static_cast<long long>(A[base + static_cast<std::size_t>(j)]) * x[static_cast<std::size_t>(j)];
            if ((j & 255) == 255) acc %= q; // 防止长循环中 acc 过大
        }
        y[static_cast<std::size_t>(i)] = red(acc, q);
    }
}

void v_add(const Vec& a, const Vec& b, Vec& c, long q) {
    for (std::size_t i = 0; i < c.size(); ++i) c[i] = add_mod(a[i], b[i], q);
}

void sv_mul(long s, const Vec& x, Vec& y, long q) {
    for (std::size_t i = 0; i < y.size(); ++i) y[i] = mul_mod(s, x[i], q);
}

void madd(const Mat& A, const Mat& B, Mat& C, long q) {
    for (std::size_t i = 0; i < C.size(); ++i) C[i] = add_mod(A[i], B[i], q);
}

void sm_mul(long s, const Mat& A, Mat& C, long q) {
    for (std::size_t i = 0; i < C.size(); ++i) C[i] = mul_mod(s, A[i], q);
}

void mm_mul(const Mat& A, const Mat& B, Mat& C, long rows, long mid, long cols, long q) {
    // 行优先矩阵乘法：C = A * B mod q。
    // 为减少模运算次数，先用 64 位整数累加，最后统一取模。
    std::vector<long long> acc(C.size(), 0);
    for (long i = 0; i < rows; ++i) {
        for (long t = 0; t < mid; ++t) {
            const long av = A[idx(mid, i, t)];
            const auto b_base = static_cast<std::size_t>(t) * static_cast<std::size_t>(cols);
            const auto c_base = static_cast<std::size_t>(i) * static_cast<std::size_t>(cols);
            for (long j = 0; j < cols; ++j) {
                acc[c_base + static_cast<std::size_t>(j)] +=
                    static_cast<long long>(av) * B[b_base + static_cast<std::size_t>(j)];
            }
        }
    }
    for (std::size_t i = 0; i < C.size(); ++i) C[i] = red(acc[i], q);
}

void gauss_sample(VecZ& y, const DGSampler& dg, std::mt19937_64& rng) {
    // 离散高斯采样：每个坐标直接从预计算 CDT 中抽取整数。
    for (auto& v : y) v = dg.sample(rng);
}

bool norm_check(const VecZ& y, long B) {
    long long s = 0;
    const long long bound = static_cast<long long>(B) * B;
    for (long v : y) {
        s += static_cast<long long>(v) * v;
        if (s > bound) return false;
    }
    return true;
}

// ------------------------- SHA-256，用于 hash-to-matrix -------------------------

inline std::uint32_t rotr(std::uint32_t x, std::uint32_t n) {
    return (x >> n) | (x << (32U - n));
}

std::array<std::uint8_t, 32> sha256(const std::vector<std::uint8_t>& msg) {
    static const std::uint32_t k[64] = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
    };

    std::vector<std::uint8_t> data = msg;
    const std::uint64_t bit_len = static_cast<std::uint64_t>(data.size()) * 8ULL;
    data.push_back(0x80U);
    while ((data.size() % 64U) != 56U) data.push_back(0);
    for (int i = 7; i >= 0; --i) data.push_back(static_cast<std::uint8_t>((bit_len >> (8 * i)) & 0xffU));

    std::uint32_t h[8] = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
    };

    for (std::size_t off = 0; off < data.size(); off += 64) {
        std::uint32_t w[64]{};
        for (int i = 0; i < 16; ++i) {
            const std::size_t p = off + static_cast<std::size_t>(4 * i);
            w[i] = (static_cast<std::uint32_t>(data[p]) << 24) |
                   (static_cast<std::uint32_t>(data[p + 1]) << 16) |
                   (static_cast<std::uint32_t>(data[p + 2]) << 8) |
                   (static_cast<std::uint32_t>(data[p + 3]));
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = hh + S1 + ch + k[i] + w[i];
            const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = S0 + maj;
            hh = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    std::array<std::uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i) {
        out[4 * i] = static_cast<std::uint8_t>((h[i] >> 24) & 0xffU);
        out[4 * i + 1] = static_cast<std::uint8_t>((h[i] >> 16) & 0xffU);
        out[4 * i + 2] = static_cast<std::uint8_t>((h[i] >> 8) & 0xffU);
        out[4 * i + 3] = static_cast<std::uint8_t>(h[i] & 0xffU);
    }
    return out;
}

void append_u64(std::vector<std::uint8_t>& v, std::uint64_t x) {
    for (int i = 7; i >= 0; --i) v.push_back(static_cast<std::uint8_t>((x >> (8 * i)) & 0xffU));
}

void hash_mat(Mat& C, long rows, long cols, long q, std::uint64_t nonce) {
    // 哈希到矩阵：用 SHA-256(seed || counter) 扩展出矩阵元素并模 q。
    const std::size_t total = checked_size(rows, cols, "hash matrix");
    std::size_t pos = 0;
    std::uint64_t ctr = 0;
    while (pos < total) {
        std::vector<std::uint8_t> in;
        const char tag[] = "NTL-lattice-hash-to-matrix";
        in.insert(in.end(), tag, tag + sizeof(tag) - 1);
        append_u64(in, nonce);
        append_u64(in, ctr++);
        const auto dg = sha256(in);
        for (std::size_t i = 0; i + 1 < dg.size() && pos < total; i += 2) {
            const long x = static_cast<long>((static_cast<unsigned>(dg[i]) << 8) | dg[i + 1]);
            C[pos++] = x % q;
        }
    }
}

template <class F>
double bench(int iters, F&& fn) {
    // 预热 2 次，减少首次执行造成的偶然扰动；预热不计入平均时间。
    for (int i = 0; i < 2; ++i) sink ^= fn();
    const auto st = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) sink ^= (fn() + static_cast<std::uint64_t>(i));
    const auto ed = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(ed - st).count();
    return ms / static_cast<double>(iters);
}

void print_result(const Result& r) {
    std::cout << std::left << std::setw(34) << r.name
              << std::right << std::setw(14) << std::fixed << std::setprecision(6)
              << r.avg_ms << " ms\n";
}

void write_csv(const std::string& path, const std::vector<Result>& rs) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("无法写入 CSV 文件: " + path);
    out << "operation,avg_ms\n";
    for (const auto& r : rs) out << r.name << "," << std::fixed << std::setprecision(9) << r.avg_ms << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Opt o = parse(argc, argv);

        // 初始化 NTL 随机种子：本程序主要用 NTL 的模运算函数，随机数据由标准库生成以减少初始化时间。
        NTL::SetSeed(NTL::conv<NTL::ZZ>(static_cast<long>(o.seed & 0x7fffffffULL)));

        std::mt19937_64 rng(o.seed);
        const long small = 3;

        // 预生成被测输入。输入生成不计入操作耗时。
        Mat A_NM(checked_size(o.N, o.M, "A_NM"));
        Mat A_NK(checked_size(o.N, o.K, "A_NK"));
        Mat B_NK(checked_size(o.N, o.K, "B_NK"));
        Mat A_MK(checked_size(o.M, o.K, "A_MK"));
        Mat B_MK(checked_size(o.M, o.K, "B_MK"));
        Mat C_KK(checked_size(o.K, o.K, "C_KK"));
        Vec x_N(static_cast<std::size_t>(o.N)), y_N(static_cast<std::size_t>(o.N)), z_N(static_cast<std::size_t>(o.N));
        Vec x_M(static_cast<std::size_t>(o.M)), y_M(static_cast<std::size_t>(o.M)), z_M(static_cast<std::size_t>(o.M));
        VecZ g(static_cast<std::size_t>(o.samp_dim));

        fill_mod(A_NM, o.q, rng);
        fill_mod(A_NK, o.q, rng);
        fill_mod(B_NK, o.q, rng);
        fill_mod(A_MK, o.q, rng);
        fill_mod(B_MK, o.q, rng);
        fill_mod(C_KK, o.q, rng);
        fill_mod(x_N, o.q, rng);
        fill_mod(y_N, o.q, rng);
        fill_mod(x_M, o.q, rng);
        fill_mod(y_M, o.q, rng);

        Mat out_NK(checked_size(o.N, o.K, "out_NK"));
        Mat out_MK(checked_size(o.M, o.K, "out_MK"));
        Mat out_KK(checked_size(o.K, o.K, "out_KK"));
        Vec out_vN(static_cast<std::size_t>(o.N));
        Vec out_vM(static_cast<std::size_t>(o.M));

        DGSampler dg(o.sigma, o.dg_tail);
        std::uint64_t nonce = o.seed;

        std::cout << "NTL general-lattice operation benchmark\n";
        std::cout << "Parameters: N=" << o.N << ", M=" << o.M << ", K=" << o.K
                  << ", q=" << o.q << ", iters=" << o.iters
                  << ", samp_dim=" << o.samp_dim
                  << ", sigma=" << o.sigma << ", dg_tail=" << dg.tail << "\n\n";
        std::cout << std::left << std::setw(34) << "Operation" << std::right << std::setw(18) << "Average time\n";
        std::cout << std::string(52, '-') << "\n";
        std::cout.flush();

        std::vector<Result> rs;
        std::ofstream csv_out;
        if (!o.csv.empty()) {
            csv_out.open(o.csv);
            if (!csv_out) throw std::runtime_error("无法写入 CSV 文件: " + o.csv);
            csv_out << "operation,avg_ms\n";
            csv_out.flush();
        }
        auto add = [&](const std::string& name, auto&& fn) {
            if (!o.only.empty() && o.only != name) return;
            Result r{name, bench(o.iters, std::forward<decltype(fn)>(fn))};
            rs.push_back(r);
            print_result(r);
            std::cout.flush();
            if (csv_out) {
                csv_out << r.name << "," << std::fixed << std::setprecision(9) << r.avg_ms << "\n";
                csv_out.flush();
            }
        };

        // 输出顺序严格对应 TABLE I 中一般格操作的顺序。
        add("T_mv^{N x M}", [&]() {
            mv_mul(A_NM, x_M, out_vN, o.N, o.M, o.q);
            return pick(out_vN);
        });

        add("T_vadd^{N}", [&]() {
            v_add(x_N, y_N, z_N, o.q);
            return pick(z_N);
        });

        add("T_vadd^{M}", [&]() {
            v_add(x_M, y_M, z_M, o.q);
            return pick(z_M);
        });

        add("T_sv^{N}", [&]() {
            sv_mul(small, x_N, z_N, o.q);
            return pick(z_N);
        });

        add("T_sv^{M}", [&]() {
            sv_mul(small, x_M, z_M, o.q);
            return pick(z_M);
        });

        add("T_sp", [&]() {
            gauss_sample(g, dg, rng);
            return pick_z(g);
        });

        add("T_norm", [&]() {
            const bool ok = norm_check(g, o.B);
            return ok ? 1ULL : 0ULL;
        });

        add("T_madd^{N x K}", [&]() {
            madd(A_NK, B_NK, out_NK, o.q);
            return pick(out_NK);
        });

        add("T_madd^{M x K}", [&]() {
            madd(A_MK, B_MK, out_MK, o.q);
            return pick(out_MK);
        });

        add("T_mm^{N x K,K x K}", [&]() {
            mm_mul(A_NK, C_KK, out_NK, o.N, o.K, o.K, o.q);
            return pick(out_NK);
        });

        add("T_mm^{M x K,K x K}", [&]() {
            mm_mul(A_MK, C_KK, out_MK, o.M, o.K, o.K, o.q);
            return pick(out_MK);
        });

        add("T_mm^{N x M,M x K}", [&]() {
            mm_mul(A_NM, A_MK, out_NK, o.N, o.M, o.K, o.q);
            return pick(out_NK);
        });

        add("T_h^{K x K}", [&]() {
            hash_mat(out_KK, o.K, o.K, o.q, nonce++);
            return pick(out_KK);
        });

        add("T_h^{N x K}", [&]() {
            hash_mat(out_NK, o.N, o.K, o.q, nonce++);
            return pick(out_NK);
        });

        add("T_sm^{N x K}", [&]() {
            sm_mul(small, A_NK, out_NK, o.q);
            return pick(out_NK);
        });

        add("T_sm^{M x K}", [&]() {
            sm_mul(small, A_MK, out_MK, o.q);
            return pick(out_MK);
        });

        std::cout << std::string(52, '-') << "\n";
        std::cout << "checksum=" << sink << "\n";

        if (!o.csv.empty()) std::cout << "CSV saved to: " << o.csv << "\n";
        if (!o.only.empty() && rs.empty()) throw std::runtime_error("未找到 --only 指定的操作: " + o.only);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
