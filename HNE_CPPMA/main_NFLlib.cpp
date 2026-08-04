#include <nfl.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NTRU_N
#define NTRU_N 512
#endif

#ifndef NTRU_Q
#define NTRU_Q 15361
#endif

#ifndef NTRU_QBITS
#define NTRU_QBITS 14
#endif

#ifndef NTRU_SIGMA_KEY
#define NTRU_SIGMA_KEY 4.0532
#endif

#ifndef NTRU_BASIS_GS_BOUND
#define NTRU_BASIS_GS_BOUND 129.7013
#endif

#ifndef NTRU_SIGMA_SIG
#define NTRU_SIGMA_SIG 165.736617183
#endif

#ifndef NTRU_BSIG2
#define NTRU_BSIG2 34034726
#endif

#ifndef NTRU_BSIG
#define NTRU_BSIG 5833.93
#endif

#ifndef NTRU_BETA_SIS
#define NTRU_BETA_SIS 11668
#endif

namespace {

const std::size_t n = static_cast<std::size_t>(NTRU_N);
const std::uint32_t q = static_cast<std::uint32_t>(NTRU_Q);
const std::size_t qbits = static_cast<std::size_t>(NTRU_QBITS);

// 参数要求 n=512, q=15361。15361 是 NFLlib 自带的 14-bit NTT-friendly 模数。
typedef nfl::poly_from_modulus<std::uint16_t, NTRU_N, NTRU_QBITS> Poly;

struct Opt {
  std::size_t iters;
  std::size_t warmup;
  double sigma_key;
  double basis_gs_bound;
  double sigma_sig;
  double tail;
  std::int32_t beta_sis;
  long double bsig2;
  long double bsig;
  std::string out;
};

struct Row {
  std::string op;
  std::string desc;
  std::string params;
  std::size_t iters;
  std::size_t warmup;
  double total_ms;
  double avg_ms;
  unsigned long long chk;
};

Opt parse(int argc, char** argv) {
  Opt o;
  o.iters = 1000;
  o.warmup = 50;
  o.sigma_key = static_cast<double>(NTRU_SIGMA_KEY);
  o.basis_gs_bound = static_cast<double>(NTRU_BASIS_GS_BOUND);
  o.sigma_sig = static_cast<double>(NTRU_SIGMA_SIG);
  o.tail = 12.0;
  o.beta_sis = static_cast<std::int32_t>(NTRU_BETA_SIS);
  o.bsig2 = static_cast<long double>(NTRU_BSIG2);
  o.bsig = static_cast<long double>(NTRU_BSIG);
  o.out = "results/ntru_nfllib_results.csv";

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto val = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + name);
      }
      return argv[++i];
    };
    if (a == "--iters" || a == "--iterations") {
      o.iters = static_cast<std::size_t>(std::stoull(val("--iters")));
    } else if (a == "--warmup") {
      o.warmup = static_cast<std::size_t>(std::stoull(val("--warmup")));
    } else if (a == "--sigma-key") {
      o.sigma_key = std::stod(val("--sigma-key"));
    } else if (a == "--basis-gs-bound") {
      o.basis_gs_bound = std::stod(val("--basis-gs-bound"));
    } else if (a == "--sigma" || a == "--sigma-sig") {
      o.sigma_sig = std::stod(val("--sigma-sig"));
    } else if (a == "--tail") {
      o.tail = std::stod(val("--tail"));
    } else if (a == "--beta-sis" || a == "--inf-bound") {
      o.beta_sis = static_cast<std::int32_t>(std::stol(val("--beta-sis")));
    } else if (a == "--bsig2") {
      o.bsig2 = static_cast<long double>(std::stold(val("--bsig2")));
      o.bsig = std::sqrt(o.bsig2);
    } else if (a == "--bsig" || a == "--l2-bound") {
      o.bsig = static_cast<long double>(std::stold(val("--bsig")));
      o.bsig2 = o.bsig * o.bsig;
    } else if (a == "--out" || a == "--output") {
      o.out = val("--out");
    } else if (a == "-h" || a == "--help") {
      std::cout
          << "Usage: ntru_bench [--iters 1000] [--warmup 50]\n"
          << "                  [--sigma-key 4.0532] [--sigma-sig 165.736617183]\n"
          << "                  [--bsig2 34034726] [--beta-sis 11668]\n"
          << "                  [--out results/ntru_nfllib_results.csv]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + a);
    }
  }
  if (o.iters == 0 || o.warmup == 0 || o.sigma_key <= 0.0 ||
      o.sigma_sig <= 0.0 || o.tail <= 0.0 || o.bsig2 <= 0.0L ||
      o.beta_sis <= 0) {
    throw std::invalid_argument("invalid benchmark parameters");
  }
  return o;
}

template <class Fn>
double ms(std::size_t cnt, Fn fn) {
  const auto t0 = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < cnt; ++i) {
    fn(i);
  }
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

std::string esc(const std::string& s) {
  if (s.find_first_of(",\"\n\r") == std::string::npos) {
    return s;
  }
  std::string r = "\"";
  for (std::size_t i = 0; i < s.size(); ++i) {
    r += s[i] == '"' ? "\"\"" : std::string(1, s[i]);
  }
  r += "\"";
  return r;
}

void mkdir_if_needed(const std::string& path) {
  if (path.empty()) {
    return;
  }
  std::string cmd = "mkdir -p \"" + path + "\"";
  const int rc = std::system(cmd.c_str());
  if (rc != 0) {
    throw std::runtime_error("cannot create directory: " + path);
  }
}

std::string dirname(const std::string& path) {
  const std::string::size_type p = path.find_last_of("/\\");
  return p == std::string::npos ? std::string() : path.substr(0, p);
}

void write_csv(const std::string& out, const std::vector<Row>& rows) {
  mkdir_if_needed(dirname(out));
  std::ofstream f(out.c_str(), std::ios::trunc);
  if (!f) {
    throw std::runtime_error("cannot open output: " + out);
  }
  f << "operation,description,iters,warmup,params,total_ms,avg_ms,checksum\n";
  f << std::fixed << std::setprecision(9);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const Row& r = rows[i];
    f << esc(r.op) << ',' << esc(r.desc) << ',' << r.iters << ','
      << r.warmup << ',' << esc(r.params) << ',' << r.total_ms << ','
      << r.avg_ms << ',' << r.chk << '\n';
  }
}

namespace shake256 {

std::uint64_t rol(std::uint64_t x, unsigned s) {
  return s == 0 ? x : ((x << s) | (x >> (64U - s)));
}

void keccakf(std::uint64_t st[25]) {
  static const std::uint64_t rc[24] = {
      0x0000000000000001ULL, 0x0000000000008082ULL,
      0x800000000000808aULL, 0x8000000080008000ULL,
      0x000000000000808bULL, 0x0000000080000001ULL,
      0x8000000080008081ULL, 0x8000000000008009ULL,
      0x000000000000008aULL, 0x0000000000000088ULL,
      0x0000000080008009ULL, 0x000000008000000aULL,
      0x000000008000808bULL, 0x800000000000008bULL,
      0x8000000000008089ULL, 0x8000000000008003ULL,
      0x8000000000008002ULL, 0x8000000000000080ULL,
      0x000000000000800aULL, 0x800000008000000aULL,
      0x8000000080008081ULL, 0x8000000000008080ULL,
      0x0000000080000001ULL, 0x8000000080008008ULL};
  static const unsigned rotc[24] = {
      1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
      27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
  static const unsigned piln[24] = {
      10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
      15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};

  for (unsigned round = 0; round < 24; ++round) {
    std::uint64_t bc[5];
    for (unsigned i = 0; i < 5; ++i) {
      bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
    }
    for (unsigned i = 0; i < 5; ++i) {
      const std::uint64_t t = bc[(i + 4) % 5] ^ rol(bc[(i + 1) % 5], 1);
      for (unsigned j = 0; j < 25; j += 5) {
        st[j + i] ^= t;
      }
    }

    std::uint64_t t = st[1];
    for (unsigned i = 0; i < 24; ++i) {
      const unsigned j = piln[i];
      const std::uint64_t tmp = st[j];
      st[j] = rol(t, rotc[i]);
      t = tmp;
    }

    for (unsigned j = 0; j < 25; j += 5) {
      for (unsigned i = 0; i < 5; ++i) {
        bc[i] = st[j + i];
      }
      for (unsigned i = 0; i < 5; ++i) {
        st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
      }
    }

    st[0] ^= rc[round];
  }
}

void absorb_block(std::uint64_t st[25], const std::uint8_t* block,
                  std::size_t len) {
  for (std::size_t i = 0; i < len; ++i) {
    st[i / 8] ^= static_cast<std::uint64_t>(block[i]) << (8U * (i % 8U));
  }
}

std::vector<std::uint8_t> xof(const std::vector<std::uint8_t>& in,
                              std::size_t out_len) {
  const std::size_t rate = 136;  // SHAKE256 rate: 1088 bits
  std::uint64_t st[25] = {};
  std::size_t off = 0;

  while (off + rate <= in.size()) {
    absorb_block(st, &in[off], rate);
    keccakf(st);
    off += rate;
  }

  std::array<std::uint8_t, 136> last = {};
  const std::size_t rem = in.size() - off;
  for (std::size_t i = 0; i < rem; ++i) {
    last[i] = in[off + i];
  }
  last[rem] ^= 0x1FU;       // SHAKE domain separation suffix
  last[rate - 1] ^= 0x80U;  // pad10*1
  absorb_block(st, last.data(), rate);
  keccakf(st);

  std::vector<std::uint8_t> out;
  out.reserve(out_len);
  while (out.size() < out_len) {
    for (std::size_t i = 0; i < rate && out.size() < out_len; ++i) {
      out.push_back(static_cast<std::uint8_t>(st[i / 8] >> (8U * (i % 8U))));
    }
    if (out.size() < out_len) {
      keccakf(st);
    }
  }
  return out;
}

}  // namespace shake256

std::string nfl_moduli() {
  std::ostringstream s;
  for (std::size_t i = 0; i < Poly::nmoduli; ++i) {
    if (i) {
      s << '*';
    }
    s << Poly::get_modulus(i);
  }
  return s.str();
}

void check_params() {
  if (n != 512U || q != 15361U || qbits != 14U) {
    throw std::runtime_error("parameters must be N=512, q=15361, qbits=14");
  }
  if (Poly::nmoduli != 1U || Poly::get_modulus(0) != q) {
    std::ostringstream s;
    s << "NFLlib modulus mismatch: expected one modulus 15361, got "
      << nfl_moduli()
      << ". Please use the native NFLlib uint16_t NTT-friendly modulus 15361; "
      << "no built-in NTT fallback is used.";
    throw std::runtime_error(s.str());
  }
}

Poly rand_poly(std::mt19937& rng) {
  std::uniform_int_distribution<std::uint32_t> dist(0, q - 1U);
  std::vector<std::uint16_t> c(n);
  for (std::size_t i = 0; i < n; ++i) {
    c[i] = static_cast<std::uint16_t>(dist(rng));
  }
  return Poly(c.begin(), c.end());
}

Poly mul_ntt(const Poly& lhs, const Poly& rhs) {
  // NFLlib 的 Poly 带有 32-byte 对齐属性。按值传参时，GCC 可能给出
  // “ABI for passing parameters with 32-byte alignment has changed”的提示。
  // 这里改为按 const 引用传入，再在函数内部复制，既避免 ABI note，
  // 又保留 NTT 原地变换不修改原始输入的语义。
  Poly a(lhs);
  Poly b(rhs);
  a.ntt_pow_phi();
  b.ntt_pow_phi();
  Poly c = a * b;
  c.invntt_pow_invphi();
  return c;
}

class Gauss {
 public:
  Gauss(double s, double tail) {
    lim_ = static_cast<int>(std::ceil(s * tail));
    if (lim_ <= 0) {
      throw std::invalid_argument("bad Gaussian bound");
    }
    std::vector<double> w;
    w.reserve(static_cast<std::size_t>(2 * lim_ + 1));
    const double den = 2.0 * s * s;
    for (int x = -lim_; x <= lim_; ++x) {
      w.push_back(std::exp(-static_cast<double>(x) * static_cast<double>(x) /
                           den));
    }
    dist_ = std::discrete_distribution<int>(w.begin(), w.end());
  }

  std::vector<std::int32_t> sample(std::mt19937& rng) {
    std::vector<std::int32_t> a(n);
    for (std::size_t i = 0; i < n; ++i) {
      a[i] = static_cast<std::int32_t>(dist_(rng) - lim_);
    }
    return a;
  }

 private:
  int lim_;
  std::discrete_distribution<int> dist_;
};

bool norm_ok(const std::vector<std::int32_t>& a, std::int32_t beta_sis,
             long double bsig2) {
  std::int32_t inf = 0;
  long double l2 = 0.0L;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::int32_t x = a[i] < 0 ? -a[i] : a[i];
    inf = std::max(inf, x);
    l2 += static_cast<long double>(a[i]) * static_cast<long double>(a[i]);
  }
  return inf <= beta_sis && l2 <= bsig2;
}

std::vector<std::uint8_t> msg(std::uint64_t ctr) {
  const std::string p = "HNEB-MAS NTRU H3 input: (ch_v,T_v)";
  std::vector<std::uint8_t> m(p.begin(), p.end());
  for (int s = 56; s >= 0; s -= 8) {
    m.push_back(static_cast<std::uint8_t>(ctr >> s));
  }
  return m;
}

Poly hash_poly(const std::vector<std::uint8_t>& m) {
  std::vector<std::uint16_t> c;
  c.reserve(n);
  std::uint64_t ctr = 0;
  const std::uint32_t limit = (65536U / q) * q;

  while (c.size() < n) {
    const std::string dom = "H3:SHAKE256:HashToPoint:Rq:q=15361:n=512";
    std::vector<std::uint8_t> in(dom.begin(), dom.end());
    for (int s = 56; s >= 0; s -= 8) {
      in.push_back(static_cast<std::uint8_t>(ctr >> s));
    }
    in.insert(in.end(), m.begin(), m.end());

    // 采用 SHAKE256 扩展输出，再用拒绝采样把字节均匀映射到 R_q 的系数。
    const std::size_t need = (n - c.size()) * 4U + 64U;
    const std::vector<std::uint8_t> out = shake256::xof(in, need);
    for (std::size_t i = 0; i + 1 < out.size() && c.size() < n; i += 2) {
      const std::uint32_t w =
          static_cast<std::uint32_t>(out[i]) |
          (static_cast<std::uint32_t>(out[i + 1]) << 8U);
      if (w < limit) {
        c.push_back(static_cast<std::uint16_t>(w % q));
      }
    }
    ++ctr;
  }

  return Poly(c.begin(), c.end());
}

unsigned long long chk_poly(const Poly& p) {
  unsigned long long h = 1469598103934665603ULL;
  const std::size_t step = n >= 16 ? n / 16 : 1;
  for (std::size_t j = 0; j < Poly::nmoduli; ++j) {
    for (std::size_t i = 0; i < n; i += step) {
      h ^= static_cast<unsigned long long>(p(j, i));
      h *= 1099511628211ULL;
    }
  }
  return h;
}

unsigned long long chk_polys(const std::vector<Poly>& ps) {
  unsigned long long h = 0;
  for (std::size_t i = 0; i < ps.size(); ++i) {
    h = h * 1315423911ULL + chk_poly(ps[i]);
  }
  return h;
}

unsigned long long chk_gaus(const std::vector<std::vector<std::int32_t> >& ps) {
  unsigned long long h = 0;
  for (std::size_t i = 0; i < ps.size(); ++i) {
    for (std::size_t j = 0; j < ps[i].size(); j += 32) {
      h = h * 1315423911ULL +
          static_cast<unsigned long long>(static_cast<std::int64_t>(ps[i][j]) +
                                          32768LL);
    }
  }
  return h;
}

bool self_test() {
  std::vector<std::uint16_t> ac(n, 0), bc(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    ac[i] = static_cast<std::uint16_t>((3 * i + 1) % 8);
    bc[i] = static_cast<std::uint16_t>((5 * i + 2) % 8);
  }
  const Poly a(ac.begin(), ac.end());
  const Poly b(bc.begin(), bc.end());
  const Poly got = mul_ntt(a, b);

  const std::int64_t modq = static_cast<std::int64_t>(q);
  std::vector<std::int64_t> exp(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const std::int64_t v =
          static_cast<std::int64_t>(ac[i]) * static_cast<std::int64_t>(bc[j]);
      const std::size_t k = i + j;
      if (k < n) {
        exp[k] += v;
      } else {
        exp[k - n] -= v;
      }
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint64_t e =
        static_cast<std::uint64_t>((exp[i] % modq + modq) % modq);
    if (got(0, i) != e) {
      return false;
    }
  }
  return true;
}

std::string params(const Opt& o) {
  std::ostringstream s;
  s << "R_q=Z_" << q << "[x]/(x^" << n << "+1)"
    << ";N=" << n
    << ";q=" << q
    << ";qbits=" << qbits
    << ";nfl_modulus=" << nfl_moduli()
    << ";backend=NFLlib-NTT"
    << ";sigma_key=" << o.sigma_key
    << ";basis_gs_bound=" << o.basis_gs_bound
    << ";sigma_sig=" << o.sigma_sig
    << ";Bsig2=" << static_cast<double>(o.bsig2)
    << ";Bsig=" << static_cast<double>(o.bsig)
    << ";beta_sis=" << o.beta_sis
    << ";H3=SHAKE256 HashToPoint"
    << ";tail=" << o.tail
    << ";seed=20260625";
#ifdef NFL_OPTIMIZED
  s << ";nfl_opt=true";
#else
  s << ";nfl_opt=false";
#endif
#ifdef NTT_AVX2
  s << ";ntt=AVX2";
#elif defined(NTT_SSE)
  s << ";ntt=SSE";
#else
  s << ";ntt=generic";
#endif
  return s.str();
}

Row row(const char* op, const char* desc, const Opt& o, const std::string& p,
        double total, unsigned long long chk) {
  Row r;
  r.op = op;
  r.desc = desc;
  r.params = p;
  r.iters = o.iters;
  r.warmup = o.warmup;
  r.total_ms = total;
  r.avg_ms = total / static_cast<double>(o.iters);
  r.chk = chk;
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Opt o = parse(argc, argv);
    check_params();
    if (!self_test()) {
      throw std::runtime_error("NTT negacyclic multiplication self-test failed");
    }

    std::mt19937 rng(20260625U);
    Gauss gs(o.sigma_sig, o.tail);
    const Poly a = rand_poly(rng);
    const Poly b = rand_poly(rng);
    std::vector<std::vector<std::uint8_t> > msgs(o.iters);
    for (std::size_t i = 0; i < o.iters; ++i) {
      msgs[i] = msg(static_cast<std::uint64_t>(i));
    }

    for (std::size_t i = 0; i < o.warmup; ++i) {
      // 预热顺序严格对应 TABLE I 中的 NTRU 部分：
      // T_{pm}, T_{pa}, T_{pgsp}, T_{pnorm}, T_{hp}。
      Poly pm = mul_ntt(a, b);
      Poly pa = a + b;
      std::vector<std::int32_t> pgsp = gs.sample(rng);
      volatile bool pnorm = norm_ok(pgsp, o.beta_sis, o.bsig2);
      Poly hp = hash_poly(msgs[i % msgs.size()]);
      (void)pm;
      (void)pa;
      (void)pnorm;
      (void)hp;
    }

    std::vector<Row> rows;
    const std::string ps = params(o);

    // 输出顺序严格对应 TABLE I 的 NTRU 部分。
    std::vector<Poly> mul_out(o.iters);
    const double t_pm =
        ms(o.iters, [&](std::size_t i) { mul_out[i] = mul_ntt(a, b); });
    rows.push_back(row("T_{pm}", "polynomial multiplication in R_q by NTT", o,
                       ps, t_pm, chk_polys(mul_out)));

    std::vector<Poly> add_out(o.iters);
    const double t_pa = ms(o.iters, [&](std::size_t i) { add_out[i] = a + b; });
    rows.push_back(row("T_{pa}", "polynomial addition in R_q", o, ps, t_pa,
                       chk_polys(add_out)));

    std::vector<std::vector<std::int32_t> > pgsp_out(o.iters);
    const double t_pgsp =
        ms(o.iters, [&](std::size_t i) { pgsp_out[i] = gs.sample(rng); });
    rows.push_back(row("T_{pgsp}", "polynomial discrete Gaussian sampling", o,
                       ps, t_pgsp, chk_gaus(pgsp_out)));

    std::vector<unsigned char> norm_out(o.iters);
    const double t_pnorm = ms(o.iters, [&](std::size_t i) {
      norm_out[i] = norm_ok(pgsp_out[i], o.beta_sis, o.bsig2) ? 1U : 0U;
    });
    unsigned long long norm_chk = 0;
    for (std::size_t i = 0; i < norm_out.size(); ++i) {
      norm_chk = norm_chk * 1315423911ULL + norm_out[i];
    }
    rows.push_back(row("T_{pnorm}", "polynomial norm checking", o, ps, t_pnorm,
                       norm_chk));

    std::vector<Poly> hp_out(o.iters);
    const double t_hp =
        ms(o.iters, [&](std::size_t i) { hp_out[i] = hash_poly(msgs[i]); });
    rows.push_back(row("T_{hp}", "SHAKE256 HashToPoint to R_q", o, ps, t_hp,
                       chk_polys(hp_out)));

    write_csv(o.out, rows);

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "NFLlib NTRU benchmark: R_q = Z_" << q << "[x]/(x^" << n
              << "+1), sigma_sig = " << o.sigma_sig
              << ", Bsig^2 = " << static_cast<double>(o.bsig2)
              << ", beta_SIS = " << o.beta_sis << '\n';
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const Row& r = rows[i];
      std::cout << r.op << " | avg = " << r.avg_ms
                << " ms | total = " << r.total_ms
                << " ms | checksum = " << r.chk << '\n';
    }
    std::cout << "CSV: " << o.out << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
