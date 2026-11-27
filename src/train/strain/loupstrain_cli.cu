// Standalone zero-Torch, zero-Python C++/CUDA trainer for Loup-S students (Proposal #1).
// Eliminates ~9.2-9.5 s of Python/PyTorch/e3nn/ASE startup, CPU neighbor-list, and eval overhead.
//
// Build: make train-cli CUDA_ARCH=sm_120   (or sm_89)
// Usage:
//   ./build/sm_120/loupstrain_cli --init init.lsm --train train.xyz \
//       [--valid valid.xyz] [--test teacher-test.xyz] [--recipe soap40.json] \
//       [--out trained.lsm] [--epochs 30] [--batch 1] [--seed 1] [--cutoff 5.5] \
//       [--center-e0] [--auto-stats] [--k-spline 64] [--eval-every 0]
//
// Model I/O (.lsm format LSM1): see tools/dev/lsm_io.py (dump / apply).

#include "loupstrain.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define CK_CUDA(call)                                                                                   \
    do {                                                                                                \
        cudaError_t _e = (call);                                                                        \
        if (_e != cudaSuccess) {                                                                        \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            std::exit(1);                                                                               \
        }                                                                                               \
    } while (0)

#define CK_LST(h, call)                                                                                 \
    do {                                                                                                \
        int _rc = (call);                                                                               \
        if (_rc != 0) {                                                                                 \
            std::fprintf(stderr, "loupstrain error %s:%d: %s\n", __FILE__, __LINE__,                    \
                         lst_last_error(h) ? lst_last_error(h) : "unknown");                            \
            std::exit(1);                                                                               \
        }                                                                                               \
    } while (0)

namespace {

using Clock = std::chrono::steady_clock;
double elapsed_s(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ============================================================================ Periodic table
int symbol_to_z(const char* sym, size_t len) {
    static const char* kSym[119] = {
        "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
        "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
        "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
        "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
        "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
        "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
    for (int z = 1; z <= 118; ++z) {
        if (std::strlen(kSym[z]) == len && std::memcmp(kSym[z], sym, len) == 0) return z;
    }
    return -1;
}

// ============================================================================ LSM1 model I/O
struct LsmModel {
    int32_t dtype = 0;  // 0 float32, 1 float64
    int32_t S = 0, C = 0, H = 0, NB = 0, hid = 0, K = 0, K3 = 0, K2 = 0, p_cut = 0, zbl_p = 0;
    double r_max = 0, prefactor = 0, act = 0, scale = 0, shift = 0, avg0 = 0, avg1 = 0;
    double zbl_a_exp = 0, zbl_a_pref = 0;
    double zbl_c[4] = {};
    std::vector<int32_t> Z;
    std::vector<double> E0;
    std::vector<double> cov_r;
    std::vector<double> bessel_w;
    int64_t param_numel[LST_N_PARAMS] = {};
    std::vector<float> params_f32;
    std::vector<double> params_f64;

    size_t total_params() const {
        size_t s = 0;
        for (int i = 0; i < LST_N_PARAMS; ++i) s += (size_t)param_numel[i];
        return s;
    }

    int z_to_sp(int z) const {
        for (int s = 0; s < S; ++s)
            if (Z[s] == z) return s;
        return -1;
    }
};

#pragma pack(push, 1)
struct LsmHeaderDisk {
    char magic[4];
    int32_t version, dtype;
    int32_t S, C, H, NB, hid, K, K3, K2, p_cut, zbl_p;
    double r_max, prefactor, act, scale, shift, avg0, avg1, zbl_a_exp, zbl_a_pref;
    double zbl_c[4];
};
#pragma pack(pop)

LsmModel load_lsm(const std::string& path) {
    std::ifstream fh(path, std::ios::binary);
    if (!fh) {
        std::fprintf(stderr, "Cannot open LSM file: %s\n", path.c_str());
        std::exit(1);
    }
    LsmHeaderDisk hdr{};
    fh.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (std::memcmp(hdr.magic, "LSM1", 4) != 0 || hdr.version != 1) {
        std::fprintf(stderr, "Invalid LSM1 header in %s\n", path.c_str());
        std::exit(1);
    }
    LsmModel m;
    m.dtype = hdr.dtype;
    m.S = hdr.S;
    m.C = hdr.C;
    m.H = hdr.H;
    m.NB = hdr.NB;
    m.hid = hdr.hid;
    m.K = hdr.K;
    m.K3 = hdr.K3;
    m.K2 = hdr.K2;
    m.p_cut = hdr.p_cut;
    m.zbl_p = hdr.zbl_p;
    m.r_max = hdr.r_max;
    m.prefactor = hdr.prefactor;
    m.act = hdr.act;
    m.scale = hdr.scale;
    m.shift = hdr.shift;
    m.avg0 = hdr.avg0;
    m.avg1 = hdr.avg1;
    m.zbl_a_exp = hdr.zbl_a_exp;
    m.zbl_a_pref = hdr.zbl_a_pref;
    for (int i = 0; i < 4; ++i) m.zbl_c[i] = hdr.zbl_c[i];
    m.Z.resize(m.S);
    m.E0.resize(m.S);
    m.cov_r.resize(m.S);
    m.bessel_w.resize(m.NB);
    fh.read(reinterpret_cast<char*>(m.Z.data()), m.S * sizeof(int32_t));
    fh.read(reinterpret_cast<char*>(m.E0.data()), m.S * sizeof(double));
    fh.read(reinterpret_cast<char*>(m.cov_r.data()), m.S * sizeof(double));
    fh.read(reinterpret_cast<char*>(m.bessel_w.data()), m.NB * sizeof(double));
    fh.read(reinterpret_cast<char*>(m.param_numel), LST_N_PARAMS * sizeof(int64_t));
    const size_t tot = m.total_params();
    if (m.dtype == 0) {
        m.params_f32.resize(tot);
        fh.read(reinterpret_cast<char*>(m.params_f32.data()), tot * sizeof(float));
    } else {
        m.params_f64.resize(tot);
        fh.read(reinterpret_cast<char*>(m.params_f64.data()), tot * sizeof(double));
    }
    if (!fh) {
        std::fprintf(stderr, "Truncated LSM file: %s\n", path.c_str());
        std::exit(1);
    }
    return m;
}

void save_lsm(const std::string& path, const LsmModel& m) {
    std::ofstream fh(path, std::ios::binary);
    if (!fh) {
        std::fprintf(stderr, "Cannot write LSM file: %s\n", path.c_str());
        std::exit(1);
    }
    LsmHeaderDisk hdr{};
    std::memcpy(hdr.magic, "LSM1", 4);
    hdr.version = 1;
    hdr.dtype = m.dtype;
    hdr.S = m.S;
    hdr.C = m.C;
    hdr.H = m.H;
    hdr.NB = m.NB;
    hdr.hid = m.hid;
    hdr.K = m.K;
    hdr.K3 = m.K3;
    hdr.K2 = m.K2;
    hdr.p_cut = m.p_cut;
    hdr.zbl_p = m.zbl_p;
    hdr.r_max = m.r_max;
    hdr.prefactor = m.prefactor;
    hdr.act = m.act;
    hdr.scale = m.scale;
    hdr.shift = m.shift;
    hdr.avg0 = m.avg0;
    hdr.avg1 = m.avg1;
    hdr.zbl_a_exp = m.zbl_a_exp;
    hdr.zbl_a_pref = m.zbl_a_pref;
    for (int i = 0; i < 4; ++i) hdr.zbl_c[i] = m.zbl_c[i];
    fh.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    fh.write(reinterpret_cast<const char*>(m.Z.data()), m.S * sizeof(int32_t));
    fh.write(reinterpret_cast<const char*>(m.E0.data()), m.S * sizeof(double));
    fh.write(reinterpret_cast<const char*>(m.cov_r.data()), m.S * sizeof(double));
    fh.write(reinterpret_cast<const char*>(m.bessel_w.data()), m.NB * sizeof(double));
    fh.write(reinterpret_cast<const char*>(m.param_numel), LST_N_PARAMS * sizeof(int64_t));
    if (m.dtype == 0) {
        fh.write(reinterpret_cast<const char*>(m.params_f32.data()), m.params_f32.size() * sizeof(float));
    } else {
        fh.write(reinterpret_cast<const char*>(m.params_f64.data()), m.params_f64.size() * sizeof(double));
    }
}

// ============================================================================ Fast extXYZ parser
struct RawFrame {
    int n = 0;
    double cell[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    int pbc[3] = {1, 1, 1};
    double volume = 1.0;
    double energy = 0.0;
    double stress[9] = {};
    bool has_stress = false;
    double w_e = 1.0, w_f = 1.0, w_s = 1.0;
    std::vector<int32_t> sp;    // [n] species index 0..S-1
    std::vector<double> pos;    // [n * 3]
    std::vector<double> force;  // [n * 3]
    std::vector<double> node_energy;  // [n] optional REF_node_energy
    bool has_node_energy = false;
};

static bool find_kv(const char* line, size_t len, const char* key, std::string& val_out) {
    const size_t klen = std::strlen(key);
    size_t i = 0;
    while (i < len) {
        while (i < len && (line[i] == ' ' || line[i] == '\t')) ++i;
        size_t kstart = i;
        while (i < len && line[i] != '=' && line[i] != ' ' && line[i] != '\t') ++i;
        size_t kend = i;
        if (i < len && line[i] == '=') {
            ++i;
            size_t vstart = i, vend = i;
            if (i < len && line[i] == '"') {
                ++i;
                vstart = i;
                while (i < len && line[i] != '"') {
                    if (line[i] == '\\' && i + 1 < len) ++i;
                    ++i;
                }
                vend = i;
                if (i < len && line[i] == '"') ++i;
            } else {
                vstart = i;
                while (i < len && line[i] != ' ' && line[i] != '\t' && line[i] != '\r' && line[i] != '\n') ++i;
                vend = i;
            }
            if (kend - kstart == klen && std::memcmp(line + kstart, key, klen) == 0) {
                val_out.assign(line + vstart, vend - vstart);
                return true;
            }
        } else {
            while (i < len && line[i] != ' ' && line[i] != '\t') ++i;
        }
    }
    return false;
}

static inline double fast_strtod(const char* p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t')) ++p;
    if (p >= end) return 0.0;
    const char* orig = p;
    bool neg = false;
    if (*p == '-') { neg = true; ++p; }
    else if (*p == '+') { ++p; }
    uint64_t mant = 0;
    int sig_digits = 0;
    int exp10 = 0;
    bool saw_dot = false, any_digit = false;
    while (p < end) {
        const unsigned d = (unsigned)(*p - '0');
        if (d < 10u) {
            any_digit = true;
            if (sig_digits < 15) {
                if (mant != 0 || d != 0) {
                    mant = mant * 10ULL + d;
                    ++sig_digits;
                }
                if (saw_dot) --exp10;
            } else {
                return std::strtod(orig, nullptr);
            }
            ++p;
        } else if (*p == '.' && !saw_dot) {
            saw_dot = true;
            ++p;
        } else {
            break;
        }
    }
    if (!any_digit) return std::strtod(orig, nullptr);
    if (p < end && (*p == 'e' || *p == 'E')) {
        ++p;
        bool eneg = false;
        if (p < end && *p == '-') { eneg = true; ++p; }
        else if (p < end && *p == '+') { ++p; }
        int e = 0;
        bool has_e = false;
        while (p < end && (unsigned)(*p - '0') < 10u) {
            e = e * 10 + (*p - '0');
            has_e = true;
            if (e > 300) return std::strtod(orig, nullptr);
            ++p;
        }
        if (!has_e) return std::strtod(orig, nullptr);
        exp10 += eneg ? -e : e;
    }
    static constexpr double kPow10[23] = {
        1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
        1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
    double v = (double)mant;
    if (exp10 > 0) {
        if (exp10 <= 22) v *= kPow10[exp10];
        else return std::strtod(orig, nullptr);
    } else if (exp10 < 0) {
        if (-exp10 <= 22) v /= kPow10[-exp10];
        else return std::strtod(orig, nullptr);
    }
    return neg ? -v : v;
}

static std::vector<double> extract_doubles(const std::string& s) {
    std::vector<double> out;
    const char* p = s.c_str();
    const char* end = p + s.size();
    while (p < end) {
        while (p < end && !(std::isdigit((unsigned char)*p) || *p == '-' || *p == '+' || *p == '.')) ++p;
        if (p >= end) break;
        char* nxt = nullptr;
        double v = std::strtod(p, &nxt);
        if (nxt == p) {
            ++p;
            continue;
        }
        out.push_back(v);
        p = nxt;
    }
    return out;
}

static double det3(const double c[9]) {
    return c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) +
           c[2] * (c[3] * c[7] - c[4] * c[6]);
}

std::vector<RawFrame> load_extxyz(const std::string& path, const LsmModel& m) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "Cannot open extXYZ: %s\n", path.c_str());
        std::exit(1);
    }
    std::fseek(fp, 0, SEEK_END);
    const size_t fsz = (size_t)std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::string buf(fsz, '\0');
    if (std::fread(&buf[0], 1, fsz, fp) != fsz) {
        std::fprintf(stderr, "Read failed on %s\n", path.c_str());
        std::exit(1);
    }
    std::fclose(fp);

    // Precompute fast species symbol lookup table for the model's S species (3 entries for SrTiO3!)
    static const char* kSym[119] = {
        "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
        "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
        "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
        "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
        "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
        "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
    struct SpEntry { const char* sym; size_t len; };
    std::vector<SpEntry> sp_lut(m.S);
    for (int s = 0; s < m.S; ++s) {
        int z = m.Z[s];
        const char* sym = (z >= 1 && z <= 118) ? kSym[z] : "X";
        sp_lut[s] = {sym, std::strlen(sym)};
    }

    const char* cur = buf.data();
    const char* end = cur + fsz;
    auto next_line_from = [end](const char*& p, const char*& lstart, size_t& llen) -> bool {
        if (p >= end) return false;
        lstart = p;
        const char* nl = static_cast<const char*>(std::memchr(p, '\n', (size_t)(end - p)));
        if (nl) {
            llen = (size_t)(nl - p);
            p = nl + 1;
        } else {
            llen = (size_t)(end - p);
            p = end;
        }
        if (llen > 0 && lstart[llen - 1] == '\r') --llen;
        return true;
    };

    // Phase 1: fast newline scan to locate frame spans (~1 ms for 300 frames)
    struct FrameSpan {
        int n;
        const char* hdr;
        size_t hdr_len;
        const char* atoms;
    };
    std::vector<FrameSpan> spans;
    spans.reserve(512);
    const char* ls = nullptr;
    size_t ll = 0;
    while (next_line_from(cur, ls, ll)) {
        while (ll > 0 && (*ls == ' ' || *ls == '\t')) { ++ls; --ll; }
        if (ll == 0) continue;
        const int n = std::atoi(ls);
        if (n <= 0) {
            std::fprintf(stderr, "Bad atom count in %s\n", path.c_str());
            std::exit(1);
        }
        const char* hs = nullptr;
        size_t hl = 0;
        if (!next_line_from(cur, hs, hl)) break;
        const char* atoms_ptr = cur;
        for (int a = 0; a < n; ++a) {
            const char* nl = static_cast<const char*>(std::memchr(cur, '\n', (size_t)(end - cur)));
            if (nl) {
                cur = nl + 1;
            } else if (a == n - 1 && cur < end) {
                cur = end;
            } else {
                std::fprintf(stderr, "Unexpected EOF reading atom %d in %s\n", a, path.c_str());
                std::exit(1);
            }
        }
        spans.push_back(FrameSpan{n, hs, hl, atoms_ptr});
    }

    // Phase 2: parallel multi-threaded frame parsing with exact fast_strtod
    const size_t NF = spans.size();
    std::vector<RawFrame> frames(NF);
    auto parse_range = [&](size_t f0, size_t f1) {
        std::string val;
        for (size_t fi = f0; fi < f1; ++fi) {
            const FrameSpan& spn = spans[fi];
            RawFrame& f = frames[fi];
            f.n = spn.n;
            const char* hl_s = spn.hdr;
            const size_t hl_l = spn.hdr_len;

            if (find_kv(hl_s, hl_l, "Lattice", val)) {
                auto c = extract_doubles(val);
                if (c.size() != 9) {
                    std::fprintf(stderr, "Bad Lattice (%zu nums) in %s\n", c.size(), path.c_str());
                    std::exit(1);
                }
                for (int i = 0; i < 9; ++i) f.cell[i] = c[i];
                f.volume = std::fabs(det3(f.cell));
            }
            if (find_kv(hl_s, hl_l, "pbc", val)) {
                int idx = 0;
                for (char ch : val) {
                    if (ch == 'T' || ch == 't' || ch == '1') {
                        if (idx < 3) f.pbc[idx++] = 1;
                    } else if (ch == 'F' || ch == 'f' || ch == '0') {
                        if (idx < 3) f.pbc[idx++] = 0;
                    }
                }
            }
            if (find_kv(hl_s, hl_l, "REF_energy", val) || find_kv(hl_s, hl_l, "energy", val)) {
                f.energy = fast_strtod(val.data(), val.data() + val.size());
            }
            if (find_kv(hl_s, hl_l, "REF_stress", val) || find_kv(hl_s, hl_l, "stress", val)) {
                auto s = extract_doubles(val);
                if (s.size() == 9) {
                    for (int i = 0; i < 9; ++i) f.stress[i] = s[i];
                    f.has_stress = true;
                } else if (s.size() == 6) {
                    f.stress[0] = s[0]; f.stress[4] = s[1]; f.stress[8] = s[2];
                    f.stress[5] = f.stress[7] = s[3];
                    f.stress[2] = f.stress[6] = s[4];
                    f.stress[1] = f.stress[3] = s[5];
                    f.has_stress = true;
                }
            }
            int col_sp = 0, col_pos = 1, col_frc = -1, col_ei = -1;
            if (find_kv(hl_s, hl_l, "Properties", val)) {
                int col = 0;
                size_t p = 0;
                while (p < val.size()) {
                    size_t c1 = val.find(':', p);
                    if (c1 == std::string::npos) break;
                    size_t c2 = val.find(':', c1 + 1);
                    if (c2 == std::string::npos) break;
                    size_t c3 = val.find(':', c2 + 1);
                    std::string name = val.substr(p, c1 - p);
                    int cnt = std::atoi(val.substr(c2 + 1, (c3 == std::string::npos ? val.size() : c3) - c2 - 1).c_str());
                    if (name == "species") col_sp = col;
                    else if (name == "pos") col_pos = col;
                    else if (name == "REF_forces" || (name == "forces" && col_frc < 0)) col_frc = col;
                    else if (name == "REF_node_energy" || (name == "node_energy" && col_ei < 0)) col_ei = col;
                    col += cnt;
                    if (c3 == std::string::npos) break;
                    p = c3 + 1;
                }
            }
            const int max_col = std::max(col_sp, std::max(col_pos + 2, std::max(col_frc >= 0 ? col_frc + 2 : 0, col_ei >= 0 ? col_ei : 0)));
            f.sp.resize(f.n);
            f.pos.resize(f.n * 3);
            f.force.assign(f.n * 3, 0.0);
            if (col_ei >= 0) {
                f.has_node_energy = true;
                f.node_energy.assign(f.n, 0.0);
            }

            const char* ap = spn.atoms;
            for (int a = 0; a < f.n; ++a) {
                const char* als = nullptr;
                size_t all = 0;
                next_line_from(ap, als, all);
                const char* p = als;
                const char* le = als + all;
                int col = 0;
                while (p < le && col <= max_col) {
                    while (p < le && (*p == ' ' || *p == '\t')) ++p;
                    if (p >= le) break;
                    const char* t0 = p;
                    while (p < le && *p != ' ' && *p != '\t') ++p;
                    const size_t tlen = (size_t)(p - t0);
                    if (col == col_sp) {
                        int sp = -1;
                        if (std::isdigit((unsigned char)t0[0])) {
                            sp = m.z_to_sp(std::atoi(t0));
                        } else {
                            for (int s = 0; s < m.S; ++s) {
                                if (sp_lut[s].len == tlen && std::memcmp(sp_lut[s].sym, t0, tlen) == 0) {
                                    sp = s;
                                    break;
                                }
                            }
                            if (sp < 0) sp = m.z_to_sp(symbol_to_z(t0, tlen));
                        }
                        if (sp < 0) {
                            std::fprintf(stderr, "Unknown species '%.*s' in %s\n", (int)tlen, t0, path.c_str());
                            std::exit(1);
                        }
                        f.sp[a] = sp;
                    } else if (col >= col_pos && col < col_pos + 3) {
                        f.pos[a * 3 + (col - col_pos)] = fast_strtod(t0, p);
                    } else if (col_frc >= 0 && col >= col_frc && col < col_frc + 3) {
                        f.force[a * 3 + (col - col_frc)] = fast_strtod(t0, p);
                    } else if (col_ei >= 0 && col == col_ei) {
                        f.node_energy[a] = fast_strtod(t0, p);
                    }
                    ++col;
                }
            }
        }
    };

    const size_t n_thr = (NF >= 32) ? std::min<size_t>(4, NF / 16) : 1;
    if (n_thr <= 1) {
        parse_range(0, NF);
    } else {
        std::vector<std::thread> thrs;
        thrs.reserve(n_thr - 1);
        const size_t chunk = (NF + n_thr - 1) / n_thr;
        for (size_t ti = 1; ti < n_thr; ++ti) {
            const size_t f0 = ti * chunk;
            const size_t f1 = std::min(NF, f0 + chunk);
            if (f0 < f1) thrs.emplace_back(parse_range, f0, f1);
        }
        parse_range(0, std::min(NF, chunk));
        for (auto& th : thrs) th.join();
    }
    return frames;
}

// ============================================================================ Exact E0 minimum-norm fit (SVD / normal eq via Jacobi)
// Fits counts[F, S] @ e0[S] ~ energies[F] with minimum Euclidean norm (matches numpy.linalg.lstsq rcond=None).
std::vector<double> fit_min_norm_e0(const std::vector<RawFrame>& frames, int S) {
    const int F = (int)frames.size();
    std::vector<double> A(F * S, 0.0), y(F, 0.0);
    for (int f = 0; f < F; ++f) {
        y[f] = frames[f].energy;
        for (int sp : frames[f].sp) A[f * S + sp] += 1.0;
    }
    // Form S x S symmetric G = A^T A and b = A^T y (S is tiny, e.g. 3..10).
    std::vector<double> G(S * S, 0.0), b(S, 0.0);
    for (int i = 0; i < S; ++i) {
        for (int f = 0; f < F; ++f) b[i] += A[f * S + i] * y[f];
        for (int j = 0; j < S; ++j) {
            double s = 0.0;
            for (int f = 0; f < F; ++f) s += A[f * S + i] * A[f * S + j];
            G[i * S + j] = s;
        }
    }
    // Jacobi eigendecomposition of symmetric S x S matrix G = V diag(lam) V^T
    std::vector<double> V(S * S, 0.0);
    for (int i = 0; i < S; ++i) V[i * S + i] = 1.0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < S; ++p)
            for (int q = p + 1; q < S; ++q) off += G[p * S + q] * G[p * S + q];
        if (off < 1e-30) break;
        for (int p = 0; p < S; ++p) {
            for (int q = p + 1; q < S; ++q) {
                double gpq = G[p * S + q];
                if (std::fabs(gpq) < 1e-18) continue;
                double th = 0.5 * std::atan2(2.0 * gpq, G[q * S + q] - G[p * S + p]);
                double c = std::cos(th), s = std::sin(th);
                for (int k = 0; k < S; ++k) {
                    double gp = G[p * S + k], gq = G[q * S + k];
                    G[p * S + k] = c * gp - s * gq;
                    G[q * S + k] = s * gp + c * gq;
                }
                for (int k = 0; k < S; ++k) {
                    double gp = G[k * S + p], gq = G[k * S + q];
                    G[k * S + p] = c * gp - s * gq;
                    G[k * S + q] = s * gp + c * gq;
                    double vp = V[k * S + p], vq = V[k * S + q];
                    V[k * S + p] = c * vp - s * vq;
                    V[k * S + q] = s * vp + c * vq;
                }
            }
        }
    }
    double lmax = 0.0;
    for (int i = 0; i < S; ++i) lmax = std::max(lmax, G[i * S + i]);
    const double tol = lmax * 1e-12;
    std::vector<double> e0(S, 0.0);
    for (int k = 0; k < S; ++k) {
        double lam = G[k * S + k];
        if (lam <= tol) continue;
        double vb = 0.0;
        for (int i = 0; i < S; ++i) vb += V[i * S + k] * b[i];
        double coef = vb / lam;
        for (int i = 0; i < S; ++i) e0[i] += coef * V[i * S + k];
    }
    return e0;
}

// ============================================================================ GPU Preprocessing Kernels
struct ZblParamsDev {
    int S, zbl_p;
    double scale, shift, a_exp, a_pref;
    double c[4];
    int32_t Z[16];
    double E0[16];
    double cov_r[16];
    double z_pow[16];
};

__device__ __forceinline__ void inv3x3_and_bounds(const double c[9], double inv[9], double h[3], double& vol) {
    // Rows a0=(c0,c1,c2), a1=(c3,c4,c5), a2=(c6,c7,c8).
    double cr0x = c[4] * c[8] - c[5] * c[7];
    double cr0y = c[5] * c[6] - c[3] * c[8];
    double cr0z = c[3] * c[7] - c[4] * c[6];
    double cr1x = c[7] * c[2] - c[8] * c[1];
    double cr1y = c[8] * c[0] - c[6] * c[2];
    double cr1z = c[6] * c[1] - c[7] * c[0];
    double cr2x = c[1] * c[5] - c[2] * c[4];
    double cr2y = c[2] * c[3] - c[0] * c[5];
    double cr2z = c[0] * c[4] - c[1] * c[3];
    double det = c[0] * cr0x + c[1] * cr0y + c[2] * cr0z;
    vol = fabs(det);
    double id = 1.0 / det;
    // inv columns are (a1xa2)/det, (a2xa0)/det, (a0xa1)/det => frac = pos @ inv
    inv[0] = cr0x * id; inv[1] = cr1x * id; inv[2] = cr2x * id;
    inv[3] = cr0y * id; inv[4] = cr1y * id; inv[5] = cr2y * id;
    inv[6] = cr0z * id; inv[7] = cr1z * id; inv[8] = cr2z * id;
    h[0] = vol / sqrt(cr0x * cr0x + cr0y * cr0y + cr0z * cr0z);
    h[1] = vol / sqrt(cr1x * cr1x + cr1y * cr1y + cr1z * cr1z);
    h[2] = vol / sqrt(cr2x * cr2x + cr2y * cr2y + cr2z * cr2z);
}

// Stage 1: wrap fractional coords + count incoming neighbors per receiver atom (1 block = 1 frame).
// Wave 32: FP32 MIC pre-filter (12 KB smem -> 8 blocks/SM, 64x FP32 vs FP64 throughput!) + exact FP64 verification!
__global__ void k_prep_wrap_and_count(
    int F, const int32_t* __restrict__ node_off, const double* __restrict__ pos_in,
    const double* __restrict__ cells, const int* __restrict__ pbcs, double r_max,
    double* __restrict__ pos_wrap, double* __restrict__ frac_wrap,
    int32_t* __restrict__ deg_out) {
    const int f = blockIdx.x;
    if (f >= F) return;
    const int n0 = node_off[f], n1 = node_off[f + 1], n = n1 - n0;
    const double* c = cells + f * 9;
    const int pbc0 = pbcs[f * 3 + 0], pbc1 = pbcs[f * 3 + 1], pbc2 = pbcs[f * 3 + 2];

    __shared__ double sc[9], sinv[9], sh[3];
    __shared__ double s_r2;
    __shared__ float s_pos_f[512 * 3];
    __shared__ float s_frac_f[512 * 3];
    if (threadIdx.x == 0) {
        double vol = 1.0;
        for (int i = 0; i < 9; ++i) sc[i] = c[i];
        inv3x3_and_bounds(sc, sinv, sh, vol);
        s_r2 = r_max * r_max;
    }
    __syncthreads();

    const bool use_sm = (n <= 512);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const double px = pos_in[(n0 + i) * 3 + 0];
        const double py = pos_in[(n0 + i) * 3 + 1];
        const double pz = pos_in[(n0 + i) * 3 + 2];
        double f0 = px * sinv[0] + py * sinv[3] + pz * sinv[6];
        double f1 = px * sinv[1] + py * sinv[4] + pz * sinv[7];
        double f2 = px * sinv[2] + py * sinv[5] + pz * sinv[8];
        if (pbc0) f0 -= floor(f0);
        if (pbc1) f1 -= floor(f1);
        if (pbc2) f2 -= floor(f2);
        const double wx = f0 * sc[0] + f1 * sc[3] + f2 * sc[6];
        const double wy = f0 * sc[1] + f1 * sc[4] + f2 * sc[7];
        const double wz = f0 * sc[2] + f1 * sc[5] + f2 * sc[8];
        frac_wrap[(n0 + i) * 3 + 0] = f0;
        frac_wrap[(n0 + i) * 3 + 1] = f1;
        frac_wrap[(n0 + i) * 3 + 2] = f2;
        pos_wrap[(n0 + i) * 3 + 0] = wx;
        pos_wrap[(n0 + i) * 3 + 1] = wy;
        pos_wrap[(n0 + i) * 3 + 2] = wz;
        if (use_sm) {
            s_frac_f[i * 3 + 0] = (float)f0; s_frac_f[i * 3 + 1] = (float)f1; s_frac_f[i * 3 + 2] = (float)f2;
            s_pos_f[i * 3 + 0] = (float)wx;  s_pos_f[i * 3 + 1] = (float)wy;  s_pos_f[i * 3 + 2] = (float)wz;
        }
    }
    __syncthreads();

    const double b0 = r_max / sh[0] + 1e-9;
    const double b1 = r_max / sh[1] + 1e-9;
    const double b2 = r_max / sh[2] + 1e-9;
    const double r2max = s_r2;
    const double* pw = pos_wrap + n0 * 3;
    const double* fw = frac_wrap + n0 * 3;
    const bool use_mic = use_sm && pbc0 && pbc1 && pbc2 && (b0 < 0.49) && (b1 < 0.49) && (b2 < 0.49);

    if (use_mic) {
        const float b0_f = (float)b0 + 2e-4f, b1_f = (float)b1 + 2e-4f, b2_f = (float)b2 + 2e-4f;
        const float r2_hi = (float)((r_max + 2e-3) * (r_max + 2e-3));
        const float r2_lo = (float)((r_max - 2e-3) * (r_max - 2e-3));
        const float sc0 = (float)sc[0], sc1 = (float)sc[1], sc2 = (float)sc[2];
        const float sc3 = (float)sc[3], sc4 = (float)sc[4], sc5 = (float)sc[5];
        const float sc6 = (float)sc[6], sc7 = (float)sc[7], sc8 = (float)sc[8];
        for (int r = threadIdx.x; r < n; r += blockDim.x) {
            const float rx_f = s_pos_f[r * 3 + 0], ry_f = s_pos_f[r * 3 + 1], rz_f = s_pos_f[r * 3 + 2];
            const float rf0_f = s_frac_f[r * 3 + 0], rf1_f = s_frac_f[r * 3 + 1], rf2_f = s_frac_f[r * 3 + 2];
            const double rx = pw[r * 3 + 0], ry = pw[r * 3 + 1], rz = pw[r * 3 + 2];
            int cnt = 0;
            for (int s = 0; s < n; ++s) {
                if (s == r) continue;
                const float df0 = rf0_f - s_frac_f[s * 3 + 0];
                const float s0f = -rintf(df0);
                if (fabsf(df0 + s0f) > b0_f) continue;
                const float df1 = rf1_f - s_frac_f[s * 3 + 1];
                const float s1f = -rintf(df1);
                if (fabsf(df1 + s1f) > b1_f) continue;
                const float df2 = rf2_f - s_frac_f[s * 3 + 2];
                const float s2f = -rintf(df2);
                if (fabsf(df2 + s2f) > b2_f) continue;
                const float dxf = (rx_f - s_pos_f[s * 3 + 0]) + s0f * sc0 + s1f * sc3 + s2f * sc6;
                const float dyf = (ry_f - s_pos_f[s * 3 + 1]) + s0f * sc1 + s1f * sc4 + s2f * sc7;
                const float dzf = (rz_f - s_pos_f[s * 3 + 2]) + s0f * sc2 + s1f * sc5 + s2f * sc8;
                const float d2f = dxf * dxf + dyf * dyf + dzf * dzf;
                if (d2f >= r2_hi) continue;
                if (d2f < r2_lo) {
                    ++cnt;
                } else {
                    const double dx = (rx - pw[s * 3 + 0]) + (double)s0f * sc[0] + (double)s1f * sc[3] + (double)s2f * sc[6];
                    const double dy = (ry - pw[s * 3 + 1]) + (double)s0f * sc[1] + (double)s1f * sc[4] + (double)s2f * sc[7];
                    const double dz = (rz - pw[s * 3 + 2]) + (double)s0f * sc[2] + (double)s1f * sc[5] + (double)s2f * sc[8];
                    if (dx * dx + dy * dy + dz * dz < r2max) ++cnt;
                }
            }
            deg_out[n0 + r] = cnt;
        }
        return;
    }

    // General exact fallback (non-PBC, small cell b >= 0.49, or n > 512)
    for (int r = threadIdx.x; r < n; r += blockDim.x) {
        const double rx = pw[r * 3 + 0], ry = pw[r * 3 + 1], rz = pw[r * 3 + 2];
        const double rf0 = fw[r * 3 + 0], rf1 = fw[r * 3 + 1], rf2 = fw[r * 3 + 2];
        int cnt = 0;
        for (int s = 0; s < n; ++s) {
            const double df0 = rf0 - fw[s * 3 + 0];
            const int lo0 = pbc0 ? (int)ceil(-df0 - b0) : 0;
            const int hi0 = pbc0 ? (int)floor(-df0 + b0) : 0;
            if (lo0 > hi0) continue;
            const double df1 = rf1 - fw[s * 3 + 1];
            const int lo1 = pbc1 ? (int)ceil(-df1 - b1) : 0;
            const int hi1 = pbc1 ? (int)floor(-df1 + b1) : 0;
            if (lo1 > hi1) continue;
            const double df2 = rf2 - fw[s * 3 + 2];
            const int lo2 = pbc2 ? (int)ceil(-df2 - b2) : 0;
            const int hi2 = pbc2 ? (int)floor(-df2 + b2) : 0;
            if (lo2 > hi2) continue;
            const double bx = rx - pw[s * 3 + 0], by = ry - pw[s * 3 + 1], bz = rz - pw[s * 3 + 2];
            for (int s0 = lo0; s0 <= hi0; ++s0) {
                for (int s1 = lo1; s1 <= hi1; ++s1) {
                    for (int s2 = lo2; s2 <= hi2; ++s2) {
                        if (s == r && s0 == 0 && s1 == 0 && s2 == 0) continue;
                        const double dx = bx + (double)s0 * sc[0] + (double)s1 * sc[3] + (double)s2 * sc[6];
                        const double dy = by + (double)s0 * sc[1] + (double)s1 * sc[4] + (double)s2 * sc[7];
                        const double dz = bz + (double)s0 * sc[2] + (double)s1 * sc[5] + (double)s2 * sc[8];
                        if (dx * dx + dy * dy + dz * dz < r2max) ++cnt;
                    }
                }
            }
        }
        deg_out[n0 + r] = cnt;
    }
}

// Stage 2: fill receiver-sorted edges (sender, receiver, vec_f64, vec_out) + exact ZBL core_e/core_f/core_s!
// Wave 34: drop s_pos_d[512*3] (26.2 -> 14.0 KB smem -> 2x occupancy!) + dim3(F, 2) x 160 threads with exact
// deterministic 2-half frame reduction in k_prep_find_partners (bit-identical PREP_CHECK!).
template <typename T>
__global__ void k_prep_fill_and_zbl(
    int F, const int32_t* __restrict__ node_off, const int32_t* __restrict__ edge_off,
    const int32_t* __restrict__ rptr, const int64_t* __restrict__ species,
    const double* __restrict__ pos_wrap, const double* __restrict__ frac_wrap,
    const double* __restrict__ cells, const int* __restrict__ pbcs, const double* __restrict__ vols,
    double r_max, ZblParamsDev zp,
    int32_t* __restrict__ sender_out, int32_t* __restrict__ receiver_out,
    double* __restrict__ vec64_out, T* __restrict__ vec_out,
    T* __restrict__ core_e_out, T* __restrict__ core_f_out, T* __restrict__ core_s_out,
    double* __restrict__ core_ei64_out = nullptr, double* __restrict__ part_es_out = nullptr) {
    const int f = blockIdx.x;
    if (f >= F) return;
    const int half = blockIdx.y, n_halves = gridDim.y;
    const int n0 = node_off[f], n1 = node_off[f + 1], n = n1 - n0;
    const int r_beg = (half * n) / n_halves;
    const int r_end = ((half + 1) * n) / n_halves;
    const int e0 = edge_off[f];
    const int rp0 = n0 + f;
    const int pbc0 = pbcs[f * 3 + 0], pbc1 = pbcs[f * 3 + 1], pbc2 = pbcs[f * 3 + 2];

    __shared__ double sc[9], sinv[9], sh[3];
    __shared__ float s_pos_f[512 * 3];
    __shared__ float s_frac_f[512 * 3];
    __shared__ int8_t s_sp[512];
    __shared__ double s_fe[16];
    __shared__ double s_fs[16 * 6];
    if (threadIdx.x == 0) {
        double vol = 1.0;
        for (int i = 0; i < 9; ++i) sc[i] = cells[f * 9 + i];
        inv3x3_and_bounds(sc, sinv, sh, vol);
    }
    const bool use_sm = (n <= 512);
    if (use_sm) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            s_pos_f[i * 3 + 0] = (float)pos_wrap[(n0 + i) * 3 + 0];
            s_pos_f[i * 3 + 1] = (float)pos_wrap[(n0 + i) * 3 + 1];
            s_pos_f[i * 3 + 2] = (float)pos_wrap[(n0 + i) * 3 + 2];
            s_frac_f[i * 3 + 0] = (float)frac_wrap[(n0 + i) * 3 + 0];
            s_frac_f[i * 3 + 1] = (float)frac_wrap[(n0 + i) * 3 + 1];
            s_frac_f[i * 3 + 2] = (float)frac_wrap[(n0 + i) * 3 + 2];
            s_sp[i] = (int8_t)species[n0 + i];
        }
    }
    __syncthreads();

    const double* pw = pos_wrap + n0 * 3;
    const double* fw = frac_wrap + n0 * 3;
    const double b0 = r_max / sh[0] + 1e-9;
    const double b1 = r_max / sh[1] + 1e-9;
    const double b2 = r_max / sh[2] + 1e-9;
    const double r2max = r_max * r_max;
    const double p = (double)zp.zbl_p;
    const double c_env0 = 0.5 * (p + 1.0) * (p + 2.0);
    const double c_env1 = p * (p + 2.0);
    const double c_env2 = 0.5 * p * (p + 1.0);
    const double c_denv = -0.5 * p * (p + 1.0) * (p + 2.0);
    const bool use_mic = use_sm && pbc0 && pbc1 && pbc2 && (b0 < 0.49) && (b1 < 0.49) && (b2 < 0.49);

    double loc_e = 0.0;
    double m_xx = 0.0, m_yy = 0.0, m_zz = 0.0, m_xy = 0.0, m_xz = 0.0, m_yz = 0.0;

    auto emit_edge_zbl = [&](int r, int s, int Z_r, double cov_r, double z_r_pow,
                             double dx, double dy, double dz, double d2,
                             int& w, double& atom_e, double& gx_sum, double& gy_sum, double& gz_sum) {
        const int eg = e0 + w;
        sender_out[eg] = s;
        receiver_out[eg] = r;
        if (vec64_out) {
            vec64_out[eg * 3 + 0] = dx;
            vec64_out[eg * 3 + 1] = dy;
            vec64_out[eg * 3 + 2] = dz;
        }
        vec_out[eg * 3 + 0] = (T)dx;
        vec_out[eg * 3 + 1] = (T)dy;
        vec_out[eg * 3 + 2] = (T)dz;
        ++w;

        const int sp_s = use_sm ? (int)s_sp[s] : (int)species[n0 + s];
        const double r_zbl = cov_r + zp.cov_r[sp_s];
        if (d2 < r_zbl * r_zbl) {
            const double dist = sqrt(d2);
            const int Z_s = zp.Z[sp_s];
            const double inv_a = (zp.z_pow[sp_s] + z_r_pow) / (zp.a_pref * 0.529);
            const double k_zz = 14.3996 * (double)Z_s * (double)Z_r;
            const double x = dist * inv_a;
            const double e0x = zp.c[0] * exp(-3.2 * x);
            const double e1x = zp.c[1] * exp(-0.9423 * x);
            const double e2x = zp.c[2] * exp(-0.4028 * x);
            const double e3x = zp.c[3] * exp(-0.2016 * x);
            const double phi = e0x + e1x + e2x + e3x;
            const double dphi_dr = (-3.2 * e0x - 0.9423 * e1x - 0.4028 * e2x - 0.2016 * e3x) * inv_a;
            const double inv_r = 1.0 / dist;
            const double v = k_zz * inv_r * phi;
            const double dv_dr = k_zz * inv_r * (dphi_dr - phi * inv_r);
            const double u = dist / r_zbl;
            double upm1 = 1.0;
            for (int k = 1; k < zp.zbl_p; ++k) upm1 *= u;
            const double up = upm1 * u;
            const double env = 1.0 - c_env0 * up + c_env1 * (up * u) - c_env2 * (up * u * u);
            const double one_m_u = 1.0 - u;
            const double denv_dr = c_denv * upm1 * (one_m_u * one_m_u) / r_zbl;
            atom_e += zp.scale * 0.5 * v * env;
            const double dedr = zp.scale * 0.5 * (dv_dr * env + v * denv_dr);
            const double fac = dedr * inv_r;
            const double gx = fac * dx, gy = fac * dy, gz = fac * dz;
            gx_sum += gx; gy_sum += gy; gz_sum += gz;
            m_xx += dx * gx; m_yy += dy * gy; m_zz += dz * gz;
            m_xy += dx * gy; m_xz += dx * gz; m_yz += dy * gz;
        }
    };

    if (use_mic) {
        const float b0_f = (float)b0 + 2e-4f, b1_f = (float)b1 + 2e-4f, b2_f = (float)b2 + 2e-4f;
        const float r2_hi = (float)((r_max + 2e-3) * (r_max + 2e-3));
        const float sc0 = (float)sc[0], sc1 = (float)sc[1], sc2 = (float)sc[2];
        const float sc3 = (float)sc[3], sc4 = (float)sc[4], sc5 = (float)sc[5];
        const float sc6 = (float)sc[6], sc7 = (float)sc[7], sc8 = (float)sc[8];
        for (int r = r_beg + threadIdx.x; r < r_end; r += blockDim.x) {
            const int sp_r = (int)s_sp[r];
            const int Z_r = zp.Z[sp_r];
            const double cov_r = zp.cov_r[sp_r];
            const double z_r_pow = zp.z_pow[sp_r];
            double atom_e = zp.E0[sp_r] + zp.shift;

            const float rx_f = s_pos_f[r * 3 + 0], ry_f = s_pos_f[r * 3 + 1], rz_f = s_pos_f[r * 3 + 2];
            const float rf0_f = s_frac_f[r * 3 + 0], rf1_f = s_frac_f[r * 3 + 1], rf2_f = s_frac_f[r * 3 + 2];
            const double rx = pw[r * 3 + 0], ry = pw[r * 3 + 1], rz = pw[r * 3 + 2];
            double gx_sum = 0.0, gy_sum = 0.0, gz_sum = 0.0;
            int w = rptr[rp0 + r];

            for (int s = 0; s < n; ++s) {
                if (s == r) continue;
                const float df0 = rf0_f - s_frac_f[s * 3 + 0];
                const float s0f = -rintf(df0);
                if (fabsf(df0 + s0f) > b0_f) continue;
                const float df1 = rf1_f - s_frac_f[s * 3 + 1];
                const float s1f = -rintf(df1);
                if (fabsf(df1 + s1f) > b1_f) continue;
                const float df2 = rf2_f - s_frac_f[s * 3 + 2];
                const float s2f = -rintf(df2);
                if (fabsf(df2 + s2f) > b2_f) continue;
                const float dxf = (rx_f - s_pos_f[s * 3 + 0]) + s0f * sc0 + s1f * sc3 + s2f * sc6;
                const float dyf = (ry_f - s_pos_f[s * 3 + 1]) + s0f * sc1 + s1f * sc4 + s2f * sc7;
                const float dzf = (rz_f - s_pos_f[s * 3 + 2]) + s0f * sc2 + s1f * sc5 + s2f * sc8;
                if (dxf * dxf + dyf * dyf + dzf * dzf >= r2_hi) continue;

                const double dx = (rx - pw[s * 3 + 0]) + (double)s0f * sc[0] + (double)s1f * sc[3] + (double)s2f * sc[6];
                const double dy = (ry - pw[s * 3 + 1]) + (double)s0f * sc[1] + (double)s1f * sc[4] + (double)s2f * sc[7];
                const double dz = (rz - pw[s * 3 + 2]) + (double)s0f * sc[2] + (double)s1f * sc[5] + (double)s2f * sc[8];
                const double d2 = dx * dx + dy * dy + dz * dz;
                if (d2 >= r2max) continue;
                emit_edge_zbl(r, s, Z_r, cov_r, z_r_pow, dx, dy, dz, d2, w, atom_e, gx_sum, gy_sum, gz_sum);
            }
            if (core_ei64_out) core_ei64_out[n0 + r] = atom_e;
            loc_e += atom_e;
            core_f_out[(n0 + r) * 3 + 0] = (T)(-2.0 * gx_sum);
            core_f_out[(n0 + r) * 3 + 1] = (T)(-2.0 * gy_sum);
            core_f_out[(n0 + r) * 3 + 2] = (T)(-2.0 * gz_sum);
        }
    } else {
        for (int r = r_beg + threadIdx.x; r < r_end; r += blockDim.x) {
            const int sp_r = use_sm ? (int)s_sp[r] : (int)species[n0 + r];
            const int Z_r = zp.Z[sp_r];
            const double cov_r = zp.cov_r[sp_r];
            const double z_r_pow = zp.z_pow[sp_r];
            double atom_e = zp.E0[sp_r] + zp.shift;

            const double rx = pw[r * 3 + 0], ry = pw[r * 3 + 1], rz = pw[r * 3 + 2];
            const double rf0 = fw[r * 3 + 0], rf1 = fw[r * 3 + 1], rf2 = fw[r * 3 + 2];
            double gx_sum = 0.0, gy_sum = 0.0, gz_sum = 0.0;
            int w = rptr[rp0 + r];

            for (int s = 0; s < n; ++s) {
                const double df0 = rf0 - fw[s * 3 + 0];
                const int lo0 = pbc0 ? (int)ceil(-df0 - b0) : 0;
                const int hi0 = pbc0 ? (int)floor(-df0 + b0) : 0;
                if (lo0 > hi0) continue;
                const double df1 = rf1 - fw[s * 3 + 1];
                const int lo1 = pbc1 ? (int)ceil(-df1 - b1) : 0;
                const int hi1 = pbc1 ? (int)floor(-df1 + b1) : 0;
                if (lo1 > hi1) continue;
                const double df2 = rf2 - fw[s * 3 + 2];
                const int lo2 = pbc2 ? (int)ceil(-df2 - b2) : 0;
                const int hi2 = pbc2 ? (int)floor(-df2 + b2) : 0;
                if (lo2 > hi2) continue;
                const double bx = rx - pw[s * 3 + 0], by = ry - pw[s * 3 + 1], bz = rz - pw[s * 3 + 2];
                for (int s0 = lo0; s0 <= hi0; ++s0) {
                    for (int s1 = lo1; s1 <= hi1; ++s1) {
                        for (int s2 = lo2; s2 <= hi2; ++s2) {
                            if (s == r && s0 == 0 && s1 == 0 && s2 == 0) continue;
                            const double dx = bx + (double)s0 * sc[0] + (double)s1 * sc[3] + (double)s2 * sc[6];
                            const double dy = by + (double)s0 * sc[1] + (double)s1 * sc[4] + (double)s2 * sc[7];
                            const double dz = bz + (double)s0 * sc[2] + (double)s1 * sc[5] + (double)s2 * sc[8];
                            const double d2 = dx * dx + dy * dy + dz * dz;
                            if (d2 >= r2max) continue;
                            emit_edge_zbl(r, s, Z_r, cov_r, z_r_pow, dx, dy, dz, d2, w, atom_e, gx_sum, gy_sum, gz_sum);
                        }
                    }
                }
            }
            if (core_ei64_out) core_ei64_out[n0 + r] = atom_e;
            loc_e += atom_e;
            core_f_out[(n0 + r) * 3 + 0] = (T)(-2.0 * gx_sum);
            core_f_out[(n0 + r) * 3 + 1] = (T)(-2.0 * gy_sum);
            core_f_out[(n0 + r) * 3 + 2] = (T)(-2.0 * gz_sum);
        }
    }

    for (int o = 16; o > 0; o >>= 1) {
        loc_e += __shfl_xor_sync(0xffffffffu, loc_e, o);
        m_xx += __shfl_xor_sync(0xffffffffu, m_xx, o);
        m_yy += __shfl_xor_sync(0xffffffffu, m_yy, o);
        m_zz += __shfl_xor_sync(0xffffffffu, m_zz, o);
        m_xy += __shfl_xor_sync(0xffffffffu, m_xy, o);
        m_xz += __shfl_xor_sync(0xffffffffu, m_xz, o);
        m_yz += __shfl_xor_sync(0xffffffffu, m_yz, o);
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) {
        s_fe[warp] = loc_e;
        s_fs[warp * 6 + 0] = m_xx;
        s_fs[warp * 6 + 1] = m_yy;
        s_fs[warp * 6 + 2] = m_zz;
        s_fs[warp * 6 + 3] = m_xy;
        s_fs[warp * 6 + 4] = m_xz;
        s_fs[warp * 6 + 5] = m_yz;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nwarps = (int)(blockDim.x >> 5);
        double fe = 0.0, fs[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int w = 0; w < nwarps; ++w) {
            fe += s_fe[w];
            for (int k = 0; k < 6; ++k) fs[k] += s_fs[w * 6 + k];
        }
        if (n_halves > 1 && part_es_out) {
            double* dst = part_es_out + (long)(f * n_halves + half) * 7;
            dst[0] = fe;
            for (int k = 0; k < 6; ++k) dst[1 + k] = fs[k];
        } else {
            core_e_out[f] = (T)fe;
            const double iv = 1.0 / vols[f];
            T* cs = core_s_out + f * 9;
            cs[0] = (T)(fs[0] * iv);
            cs[4] = (T)(fs[1] * iv);
            cs[8] = (T)(fs[2] * iv);
            cs[1] = cs[3] = (T)(fs[3] * iv);
            cs[2] = cs[6] = (T)(fs[4] * iv);
            cs[5] = cs[7] = (T)(fs[5] * iv);
        }
    }
}

// Stage 3: O(deg) exact reverse-edge pairing per frame (reads vec_out directly: skips 332 MB d_vec64 in FP32!).
// Also finishes deterministic 2-half reduction of core_e / core_s when part_es_in != nullptr.
template <typename T>
__global__ void k_prep_find_partners(
    int F, const int32_t* __restrict__ node_off, const int32_t* __restrict__ edge_off,
    const int32_t* __restrict__ rptr, const int32_t* __restrict__ sender,
    const int32_t* __restrict__ receiver, const T* __restrict__ vec,
    int32_t* __restrict__ partner_out, int32_t* __restrict__ err_flag,
    const double* __restrict__ part_es_in = nullptr, int n_halves = 1,
    const double* __restrict__ vols = nullptr, T* __restrict__ core_e_out = nullptr,
    T* __restrict__ core_s_out = nullptr) {
    const int f = blockIdx.x;
    if (f >= F) return;
    if (part_es_in && n_halves > 1 && threadIdx.x == 0) {
        double fe = 0.0, fs[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int h = 0; h < n_halves; ++h) {
            const double* src = part_es_in + (long)(f * n_halves + h) * 7;
            fe += src[0];
            for (int k = 0; k < 6; ++k) fs[k] += src[1 + k];
        }
        core_e_out[f] = (T)fe;
        const double iv = 1.0 / vols[f];
        T* cs = core_s_out + f * 9;
        cs[0] = (T)(fs[0] * iv);
        cs[4] = (T)(fs[1] * iv);
        cs[8] = (T)(fs[2] * iv);
        cs[1] = cs[3] = (T)(fs[3] * iv);
        cs[2] = cs[6] = (T)(fs[4] * iv);
        cs[5] = cs[7] = (T)(fs[5] * iv);
    }
    const int n0 = node_off[f];
    const int e0 = edge_off[f], e1 = edge_off[f + 1], E = e1 - e0;
    const int rp0 = n0 + f;
    const T tol2 = std::is_same<T, float>::value ? T(1e-6) : T(1e-22);
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const int eg = e0 + e;
        const int s = sender[eg];
        const int r = receiver[eg];
        const T vx = vec[eg * 3 + 0];
        const T vy = vec[eg * 3 + 1];
        const T vz = vec[eg * 3 + 2];
        const int sb = rptr[rp0 + s];
        const int se = rptr[rp0 + s + 1];
        int match = -1;
        for (int q = sb; q < se; ++q) {
            const int qg = e0 + q;
            if (sender[qg] != r) continue;
            const T sx = vec[qg * 3 + 0] + vx;
            const T sy = vec[qg * 3 + 1] + vy;
            const T sz = vec[qg * 3 + 2] + vz;
            if (sx * sx + sy * sy + sz * sz < tol2) {
                match = q;
                break;
            }
        }
        if (match < 0 || match == e) {
            atomicExch(err_flag, 1);
            match = e;
        }
        partner_out[eg] = match;
    }
}

// Stage 4: build pidx, pe0, pe1 per frame from partner[] (single warp/block prefix scan over E edges).
__global__ void k_prep_build_pairs(
    int F, const int32_t* __restrict__ edge_off, const int32_t* __restrict__ pair_off,
    const int32_t* __restrict__ partner,
    int32_t* __restrict__ pidx, int32_t* __restrict__ pe0, int32_t* __restrict__ pe1) {
    const int f = blockIdx.x;
    if (f >= F) return;
    const int e0 = edge_off[f], E = edge_off[f + 1] - e0;
    const int p0 = pair_off[f];
    __shared__ int s_cnt[256];
    __shared__ int s_base;
    if (threadIdx.x == 0) s_base = 0;
    __syncthreads();

    for (int base = 0; base < E; base += blockDim.x) {
        const int base_pid = s_base;  // capture before scan barriers so warp 7 cannot race s_base update!
        const int e = base + threadIdx.x;
        int is_c = 0;
        int pt = -1;
        if (e < E) {
            pt = partner[e0 + e];
            is_c = (e < pt) ? 1 : 0;
        }
        s_cnt[threadIdx.x] = is_c;
        __syncthreads();
        // Inclusive scan in shared memory
        for (int off = 1; off < (int)blockDim.x; off <<= 1) {
            int v = (threadIdx.x >= off) ? s_cnt[threadIdx.x - off] : 0;
            __syncthreads();
            s_cnt[threadIdx.x] += v;
            __syncthreads();
        }
        if (is_c) {
            const int pid = base_pid + s_cnt[threadIdx.x] - 1;
            pidx[e0 + e] = pid;
            pidx[e0 + pt] = pid;
            pe0[p0 + pid] = e;
            pe1[p0 + pid] = pt;
        }
        if (threadIdx.x == blockDim.x - 1) s_base = base_pid + s_cnt[threadIdx.x];
        __syncthreads();
    }
}

// Wave 33: 256-thread 4-way atom-split FP32-accum feature sum for exact 80-param Mode-1 VarPro (64x ALU vs FP64: 17 us -> ~1.5 us!)
template <typename T>
__global__ void __launch_bounds__(256) k_varpro_frame_sums(
    int i0, int k_valid, int C, int hid,
    const int32_t* __restrict__ batch_n,
    const T* __restrict__ e_batch,
    const T* __restrict__ f0, const T* __restrict__ tt,
    double* __restrict__ e_all, double* __restrict__ s0_all, double* __restrict__ s1_all) {
    const int j = blockIdx.x;
    if (j >= k_valid) return;
    int r0 = 0;
    for (int q = 0; q < j; ++q) r0 += batch_n[q];
    const int na = batch_n[j];
    const int tid = threadIdx.x;
    const int f = i0 + j;
    if (tid == 0) e_all[f] = (double)e_batch[j];
    __shared__ T sm0[4][64];
    __shared__ T sm1[4][64];
    const int g = tid >> 6, c = tid & 63;  // 4 groups x 64 cols
    T acc0 = T(0);
    if (c < C) {
        for (int a = g; a < na; a += 4) acc0 += f0[(size_t)(r0 + a) * C + c];
    }
    sm0[g][c] = acc0;
    T acc1 = T(0);
    if (c < hid) {
        for (int a = g; a < na; a += 4) acc1 += tt[(size_t)(r0 + a) * hid + c];
    }
    sm1[g][c] = acc1;
    __syncthreads();
    if (g == 0) {
        if (c < C) s0_all[(size_t)f * C + c] = (double)sm0[0][c] + (double)sm0[1][c] + (double)sm0[2][c] + (double)sm0[3][c];
        if (c < hid) s1_all[(size_t)f * hid + c] = (double)sm1[0][c] + (double)sm1[1][c] + (double)sm1[2][c] + (double)sm1[3][c];
    }
}

// Float64 Cholesky SPD solver: solves A x = b in-place (A is n x n symmetric positive definite, row-major)
static bool solve_spd_cholesky(int n, std::vector<double>& A, std::vector<double>& b, std::vector<double>& x) {
    std::vector<double> L((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[(size_t)i * n + j];
            for (int k = 0; k < j; ++k) s -= L[(size_t)i * n + k] * L[(size_t)j * n + k];
            if (i == j) {
                if (s <= 1e-30) return false;
                L[(size_t)i * n + j] = std::sqrt(s);
            } else {
                L[(size_t)i * n + j] = s / L[(size_t)j * n + j];
            }
        }
    }
    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= L[(size_t)i * n + k] * y[k];
        y[i] = s / L[(size_t)i * n + i];
    }
    x.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double s = y[i];
        for (int k = i + 1; k < n; ++k) s -= L[(size_t)k * n + i] * x[k];
        x[i] = s / L[(size_t)i * n + i];
    }
    return true;
}

// ============================================================================ GPU Frame Set
struct GpuFrameSet {
    int F = 0;
    int N_tot = 0, E_tot = 0, P_tot = 0;
    std::vector<int32_t> n_per, e_per;
    std::vector<int32_t> node_off_h, edge_off_h, pair_off_h;
    std::vector<double> vols_h, e_ref_h, s_ref_h;
    std::vector<double> f_ref_sorted_h;  // [N_tot, 3] in species-sorted order

    // Device arrays (owned)
    void *d_vectors = nullptr, *d_f_ref = nullptr, *d_core_f = nullptr;
    void *d_e_ref = nullptr, *d_core_e = nullptr, *d_volume = nullptr, *d_n_atoms = nullptr;
    void *d_w_e = nullptr, *d_w_f = nullptr, *d_w_s = nullptr;
    void *d_s_ref = nullptr, *d_core_s = nullptr;
    int64_t* d_species = nullptr;
    int32_t *d_node_off = nullptr, *d_edge_off = nullptr, *d_pair_off = nullptr;
    int32_t *d_sender = nullptr, *d_receiver = nullptr, *d_rptr = nullptr;
    int32_t *d_pidx = nullptr, *d_pe0 = nullptr, *d_pe1 = nullptr;
    int32_t* d_node_order = nullptr;
    void* d_ei_ref = nullptr;   // [N_tot + 1] float64-centered per-atom target MLIP local energy (Wave 30)
    bool has_ei_ref = false;

    LstFrames desc{};
    double avg_num_neighbors = 0.0;

    void free_all() {
        auto cf = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        cf(d_vectors); cf(d_f_ref); cf(d_core_f);
        cf(d_e_ref); cf(d_core_e); cf(d_volume); cf(d_n_atoms);
        cf(d_w_e); cf(d_w_f); cf(d_w_s); cf(d_s_ref); cf(d_core_s);
        cf(d_species); cf(d_node_off); cf(d_edge_off); cf(d_pair_off);
        cf(d_sender); cf(d_receiver); cf(d_rptr);
        cf(d_pidx); cf(d_pe0); cf(d_pe1); cf(d_node_order); cf(d_ei_ref);
    }
    ~GpuFrameSet() { free_all(); }
};

template <typename T>
void build_gpu_frames_t(const std::vector<RawFrame>& raw, const LsmModel& m, GpuFrameSet& g, cudaStream_t st) {
    const int F = (int)raw.size();
    g.F = F;
    g.n_per.resize(F);
    g.node_off_h.resize(F + 1, 0);
    g.vols_h.resize(F);
    g.e_ref_h.resize(F);
    g.s_ref_h.resize(F * 9);
    for (int f = 0; f < F; ++f) {
        g.n_per[f] = raw[f].n;
        g.node_off_h[f + 1] = g.node_off_h[f] + raw[f].n;
        g.vols_h[f] = raw[f].volume;
        g.e_ref_h[f] = raw[f].energy;
        for (int k = 0; k < 9; ++k) g.s_ref_h[f * 9 + k] = raw[f].stress[k];
    }
    const int N_tot = g.node_off_h[F];
    g.N_tot = N_tot;

    // Species stable-sort per frame on CPU (~1 ms for 390 frames)
    std::vector<int64_t> sp_h(N_tot + 1, 0);
    std::vector<double> pos64_h(N_tot * 3);
    std::vector<T> fref_h((N_tot + 1) * 3, (T)0);
    g.f_ref_sorted_h.resize(N_tot * 3);
    std::vector<int32_t> norder_h(N_tot);
    std::vector<double> cells_h(F * 9);
    std::vector<int> pbcs_h(F * 3);
    bool any_ei = false;
    for (int f = 0; f < F; ++f)
        if (raw[f].has_node_energy) { any_ei = true; break; }
    std::vector<double> ei_raw64_h(any_ei ? N_tot : 0, 0.0);

    std::vector<int> ord;
    for (int f = 0; f < F; ++f) {
        const auto& rf = raw[f];
        const int n = rf.n, n0 = g.node_off_h[f];
        ord.resize(n);
        std::iota(ord.begin(), ord.end(), 0);
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return rf.sp[a] < rf.sp[b]; });
        for (int j = 0; j < n; ++j) {
            const int orig = ord[j];
            norder_h[n0 + orig] = j;  // inv[orig] = sorted index j (matches data.py _inverse(order)!)
            sp_h[n0 + j] = (int64_t)rf.sp[orig];
            if (any_ei && rf.has_node_energy) ei_raw64_h[n0 + j] = rf.node_energy[orig];
            for (int k = 0; k < 3; ++k) {
                pos64_h[(n0 + j) * 3 + k] = rf.pos[orig * 3 + k];
                double fv = rf.force[orig * 3 + k];
                g.f_ref_sorted_h[(n0 + j) * 3 + k] = fv;
                fref_h[(n0 + j) * 3 + k] = (T)fv;
            }
        }
        for (int k = 0; k < 9; ++k) cells_h[f * 9 + k] = rf.cell[k];
        for (int k = 0; k < 3; ++k) pbcs_h[f * 3 + k] = rf.pbc[k];
    }

    // Upload node/cell buffers and run GPU neighbor count
    double *d_pos = nullptr, *d_posw = nullptr, *d_fracw = nullptr, *d_cells = nullptr, *d_vols64 = nullptr;
    int* d_pbcs = nullptr;
    int32_t* d_deg = nullptr;
    CK_CUDA(cudaMalloc(&g.d_node_off, (F + 1) * sizeof(int32_t)));
    CK_CUDA(cudaMalloc(&g.d_species, (N_tot + 1) * sizeof(int64_t)));
    CK_CUDA(cudaMalloc(&g.d_node_order, N_tot * sizeof(int32_t)));
    CK_CUDA(cudaMalloc(&d_pos, N_tot * 3 * sizeof(double)));
    CK_CUDA(cudaMalloc(&d_posw, N_tot * 3 * sizeof(double)));
    CK_CUDA(cudaMalloc(&d_fracw, N_tot * 3 * sizeof(double)));
    CK_CUDA(cudaMalloc(&d_cells, F * 9 * sizeof(double)));
    CK_CUDA(cudaMalloc(&d_vols64, F * sizeof(double)));
    CK_CUDA(cudaMalloc(&d_pbcs, F * 3 * sizeof(int)));
    CK_CUDA(cudaMalloc(&d_deg, N_tot * sizeof(int32_t)));

    CK_CUDA(cudaMemcpyAsync(g.d_node_off, g.node_off_h.data(), (F + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(g.d_species, sp_h.data(), (N_tot + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(g.d_node_order, norder_h.data(), N_tot * sizeof(int32_t), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(d_pos, pos64_h.data(), N_tot * 3 * sizeof(double), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(d_cells, cells_h.data(), F * 9 * sizeof(double), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(d_vols64, g.vols_h.data(), F * sizeof(double), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(d_pbcs, pbcs_h.data(), F * 3 * sizeof(int), cudaMemcpyHostToDevice, st));

    int max_n_prep = 0;
    for (int f = 0; f < F; ++f) max_n_prep = std::max(max_n_prep, g.n_per[f]);
    const int prep_tpb = std::min(256, std::max(128, ((max_n_prep + 31) / 32) * 32));
    k_prep_wrap_and_count<<<F, prep_tpb, 0, st>>>(F, g.d_node_off, d_pos, d_cells, d_pbcs, m.r_max, d_posw, d_fracw, d_deg);
    CK_CUDA(cudaGetLastError());

    std::vector<int32_t> deg_h(N_tot);
    CK_CUDA(cudaMemcpyAsync(deg_h.data(), d_deg, N_tot * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
    CK_CUDA(cudaStreamSynchronize(st));

    // Build rptr, edge_off, pair_off on CPU (< 0.3 ms)
    g.e_per.resize(F);
    g.edge_off_h.resize(F + 1, 0);
    g.pair_off_h.resize(F + 1, 0);
    std::vector<int32_t> rptr_h(N_tot + F, 0);
    int64_t nb_sum = 0, nb_cnt = 0;
    for (int f = 0; f < F; ++f) {
        const int n = g.n_per[f], n0 = g.node_off_h[f], rp0 = n0 + f;
        int cur = 0;
        rptr_h[rp0] = 0;
        for (int r = 0; r < n; ++r) {
            int d = deg_h[n0 + r];
            cur += d;
            rptr_h[rp0 + r + 1] = cur;
            if (d > 0) {
                nb_sum += d;
                ++nb_cnt;
            }
        }
        if (cur % 2 != 0) {
            std::fprintf(stderr, "Frame %d has odd edge count %d (non-reciprocal neighbors)\n", f, cur);
            std::exit(1);
        }
        g.e_per[f] = cur;
        g.edge_off_h[f + 1] = g.edge_off_h[f] + cur;
        g.pair_off_h[f + 1] = g.pair_off_h[f] + cur / 2;
    }
    g.E_tot = g.edge_off_h[F];
    g.P_tot = g.pair_off_h[F];
    g.avg_num_neighbors = nb_cnt > 0 ? (double)nb_sum / (double)nb_cnt : 1.0;

    // Allocate final frame buffers
    CK_CUDA(cudaMalloc(&g.d_edge_off, (F + 1) * sizeof(int32_t)));
    CK_CUDA(cudaMalloc(&g.d_pair_off, (F + 1) * sizeof(int32_t)));
    CK_CUDA(cudaMalloc(&g.d_rptr, (N_tot + F) * sizeof(int32_t)));
    CK_CUDA(cudaMalloc(&g.d_sender, (size_t)std::max(1, g.E_tot) * sizeof(int32_t)));
    CK_CUDA(cudaMalloc(&g.d_receiver, (size_t)std::max(1, g.E_tot) * sizeof(int32_t)));
    CK_CUDA(cudaMalloc(&g.d_pidx, (size_t)std::max(1, g.E_tot) * sizeof(int32_t)));
    CK_CUDA(cudaMalloc(&g.d_pe0, (size_t)std::max(1, g.P_tot) * sizeof(int32_t)));
    CK_CUDA(cudaMalloc(&g.d_pe1, (size_t)std::max(1, g.P_tot) * sizeof(int32_t)));
    CK_CUDA(cudaMalloc(&g.d_vectors, (size_t)(g.E_tot + 1) * 3 * sizeof(T)));
    CK_CUDA(cudaMalloc(&g.d_f_ref, (size_t)(N_tot + 1) * 3 * sizeof(T)));
    CK_CUDA(cudaMalloc(&g.d_core_f, (size_t)(N_tot + 1) * 3 * sizeof(T)));
    CK_CUDA(cudaMalloc(&g.d_e_ref, (size_t)(F + 1) * sizeof(T)));
    CK_CUDA(cudaMalloc(&g.d_core_e, (size_t)(F + 1) * sizeof(T)));
    CK_CUDA(cudaMalloc(&g.d_volume, (size_t)(F + 1) * sizeof(T)));
    CK_CUDA(cudaMalloc(&g.d_n_atoms, (size_t)(F + 1) * sizeof(T)));
    CK_CUDA(cudaMalloc(&g.d_w_e, (size_t)(F + 1) * sizeof(T)));
    CK_CUDA(cudaMalloc(&g.d_w_f, (size_t)(F + 1) * sizeof(T)));
    CK_CUDA(cudaMalloc(&g.d_w_s, (size_t)(F + 1) * sizeof(T)));
    CK_CUDA(cudaMalloc(&g.d_s_ref, (size_t)(F + 1) * 9 * sizeof(T)));
    CK_CUDA(cudaMalloc(&g.d_core_s, (size_t)(F + 1) * 9 * sizeof(T)));

    CK_CUDA(cudaMemcpyAsync(g.d_edge_off, g.edge_off_h.data(), (F + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(g.d_pair_off, g.pair_off_h.data(), (F + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(g.d_rptr, rptr_h.data(), (N_tot + F) * sizeof(int32_t), cudaMemcpyHostToDevice, st));

    // Per-frame scalars + dummy trailing row
    std::vector<T> er_h(F + 1, (T)0), vol_h(F + 1, (T)1), na_h(F + 1, (T)1);
    std::vector<T> we_h(F + 1, (T)0), wf_h(F + 1, (T)0), ws_h(F + 1, (T)0);
    std::vector<T> sr_h((F + 1) * 9, (T)0);
    for (int f = 0; f < F; ++f) {
        er_h[f] = (T)raw[f].energy;
        vol_h[f] = (T)raw[f].volume;
        na_h[f] = (T)raw[f].n;
        we_h[f] = (T)raw[f].w_e;
        wf_h[f] = (T)raw[f].w_f;
        ws_h[f] = (T)(raw[f].has_stress ? raw[f].w_s : 0.0);
        for (int k = 0; k < 9; ++k) sr_h[f * 9 + k] = (T)raw[f].stress[k];
    }
    CK_CUDA(cudaMemcpyAsync(g.d_f_ref, fref_h.data(), (N_tot + 1) * 3 * sizeof(T), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(g.d_e_ref, er_h.data(), (F + 1) * sizeof(T), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(g.d_volume, vol_h.data(), (F + 1) * sizeof(T), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(g.d_n_atoms, na_h.data(), (F + 1) * sizeof(T), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(g.d_w_e, we_h.data(), (F + 1) * sizeof(T), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(g.d_w_f, wf_h.data(), (F + 1) * sizeof(T), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(g.d_w_s, ws_h.data(), (F + 1) * sizeof(T), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(g.d_s_ref, sr_h.data(), (F + 1) * 9 * sizeof(T), cudaMemcpyHostToDevice, st));

    // Zero dummy tail rows for core_e, core_f, core_s and set far vector for dummy edge row
    CK_CUDA(cudaMemsetAsync((T*)g.d_core_e + F, 0, sizeof(T), st));
    CK_CUDA(cudaMemsetAsync((T*)g.d_core_f + N_tot * 3, 0, 3 * sizeof(T), st));
    CK_CUDA(cudaMemsetAsync((T*)g.d_core_s + F * 9, 0, 9 * sizeof(T), st));
    T far_vec[3] = {(T)(2.0 * m.r_max + 1.0), (T)0, (T)0};
    CK_CUDA(cudaMemcpyAsync((T*)g.d_vectors + (size_t)g.E_tot * 3, far_vec, 3 * sizeof(T), cudaMemcpyHostToDevice, st));

    // Stage 2: fill edges + ZBL core (Wave 34: 14 KB smem + 2 blocks/frame x 160 threads -> 2x SM occupancy!)
    double* d_core_ei64 = nullptr;
    double* d_part_es = nullptr;
    int32_t *d_partner = nullptr, *d_err = nullptr;
    const int fill_halves = (max_n_prep >= 64) ? 2 : 1;
    const int fill_tpb = (fill_halves == 2)
                             ? std::min(256, std::max(128, ((((max_n_prep + 1) / 2) + 31) / 32) * 32))
                             : std::min(256, prep_tpb);
    if (any_ei) CK_CUDA(cudaMalloc(&d_core_ei64, (size_t)std::max(1, N_tot) * sizeof(double)));
    if (fill_halves > 1) CK_CUDA(cudaMalloc(&d_part_es, (size_t)F * fill_halves * 7 * sizeof(double)));
    CK_CUDA(cudaMalloc(&d_partner, (size_t)std::max(1, g.E_tot) * sizeof(int32_t)));
    CK_CUDA(cudaMalloc(&d_err, sizeof(int32_t)));
    CK_CUDA(cudaMemsetAsync(d_err, 0, sizeof(int32_t), st));

    ZblParamsDev zp{};
    zp.S = m.S;
    zp.zbl_p = m.zbl_p;
    zp.scale = m.scale;
    zp.shift = m.shift;
    zp.a_exp = m.zbl_a_exp;
    zp.a_pref = m.zbl_a_pref;
    for (int i = 0; i < 4; ++i) zp.c[i] = m.zbl_c[i];
    for (int s = 0; s < m.S && s < 16; ++s) {
        zp.Z[s] = m.Z[s];
        zp.E0[s] = m.E0[s];
        zp.cov_r[s] = m.cov_r[s];
        zp.z_pow[s] = std::pow((double)m.Z[s], m.zbl_a_exp);
    }

    k_prep_fill_and_zbl<T><<<dim3(F, fill_halves), fill_tpb, 0, st>>>(
        F, g.d_node_off, g.d_edge_off, g.d_rptr, g.d_species, d_posw, d_fracw, d_cells, d_pbcs, d_vols64,
        m.r_max, zp, g.d_sender, g.d_receiver, nullptr, (T*)g.d_vectors,
        (T*)g.d_core_e, (T*)g.d_core_f, (T*)g.d_core_s, d_core_ei64, d_part_es);
    CK_CUDA(cudaGetLastError());

    // Stage 3 & 4: exact reverse-edge pairing + deterministic core_e/core_s reduction + pidx/pe0/pe1
    k_prep_find_partners<T><<<F, 256, 0, st>>>(
        F, g.d_node_off, g.d_edge_off, g.d_rptr, g.d_sender, g.d_receiver, (const T*)g.d_vectors, d_partner, d_err,
        d_part_es, fill_halves, d_vols64, (T*)g.d_core_e, (T*)g.d_core_s);
    CK_CUDA(cudaGetLastError());
    k_prep_build_pairs<<<F, 256, 0, st>>>(
        F, g.d_edge_off, g.d_pair_off, d_partner, g.d_pidx, g.d_pe0, g.d_pe1);
    CK_CUDA(cudaGetLastError());

    int32_t err_h = 0;
    std::vector<double> core_ei64_h(any_ei ? N_tot : 0, 0.0);
    if (any_ei) {
        CK_CUDA(cudaMemcpyAsync(core_ei64_h.data(), d_core_ei64, (size_t)N_tot * sizeof(double), cudaMemcpyDeviceToHost, st));
    }
    CK_CUDA(cudaMemcpyAsync(&err_h, d_err, sizeof(int32_t), cudaMemcpyDeviceToHost, st));
    CK_CUDA(cudaStreamSynchronize(st));
    if (err_h != 0) {
        std::fprintf(stderr, "Reverse-edge pairing failed on at least one edge!\n");
        std::exit(1);
    }

    if (any_ei) {
        // Compute float64 gauge-centered per-atom target MLIP energy per (frame, species) to preserve full fp32 precision!
        std::vector<T> ei_tgt_h(N_tot + 1, (T)0);
        std::vector<double> sp_sum(m.S, 0.0);
        std::vector<int> sp_cnt(m.S, 0);
        for (int f = 0; f < F; ++f) {
            if (!raw[f].has_node_energy) continue;
            const int n0 = g.node_off_h[f], n = g.n_per[f];
            std::fill(sp_sum.begin(), sp_sum.end(), 0.0);
            std::fill(sp_cnt.begin(), sp_cnt.end(), 0);
            for (int j = 0; j < n; ++j) {
                const int sp = (int)sp_h[n0 + j];
                sp_sum[sp] += (ei_raw64_h[n0 + j] - core_ei64_h[n0 + j]);
                sp_cnt[sp] += 1;
            }
            for (int s = 0; s < m.S; ++s)
                if (sp_cnt[s] > 0) sp_sum[s] /= (double)sp_cnt[s];
            for (int j = 0; j < n; ++j) {
                const int sp = (int)sp_h[n0 + j];
                ei_tgt_h[n0 + j] = (T)((ei_raw64_h[n0 + j] - core_ei64_h[n0 + j]) - sp_sum[sp]);
            }
        }
        CK_CUDA(cudaMalloc(&g.d_ei_ref, (size_t)(N_tot + 1) * sizeof(T)));
        CK_CUDA(cudaMemcpyAsync(g.d_ei_ref, ei_tgt_h.data(), (size_t)(N_tot + 1) * sizeof(T), cudaMemcpyHostToDevice, st));
        CK_CUDA(cudaStreamSynchronize(st));
        g.has_ei_ref = true;
    }

    cudaFree(d_pos); cudaFree(d_posw); cudaFree(d_fracw); cudaFree(d_cells);
    cudaFree(d_vols64); cudaFree(d_pbcs); cudaFree(d_deg);
    if (d_core_ei64) cudaFree(d_core_ei64); if (d_part_es) cudaFree(d_part_es); cudaFree(d_partner); cudaFree(d_err);

    g.desc.vectors = g.d_vectors;
    g.desc.species = g.d_species;
    g.desc.f_ref = g.d_f_ref;
    g.desc.core_f = g.d_core_f;
    g.desc.e_ref = g.d_e_ref;
    g.desc.core_e = g.d_core_e;
    g.desc.volume = g.d_volume;
    g.desc.n_atoms = g.d_n_atoms;
    g.desc.w_e = g.d_w_e;
    g.desc.w_f = g.d_w_f;
    g.desc.w_s = g.d_w_s;
    g.desc.s_ref = g.d_s_ref;
    g.desc.core_s = g.d_core_s;
    g.desc.n_frames = F;
    g.desc.dummy_row = 32;
    g.desc.node_off = g.d_node_off;
    g.desc.edge_off = g.d_edge_off;
    g.desc.pair_off = g.d_pair_off;
    g.desc.sender = g.d_sender;
    g.desc.receiver = g.d_receiver;
    g.desc.receiver_ptr = g.d_rptr;
    g.desc.pidx = g.d_pidx;
    g.desc.pe0 = g.d_pe0;
    g.desc.pe1 = g.d_pe1;
}

LstShape compute_shape(const GpuFrameSet& g, int B) {
    const int DUMMY_ROW = 32;
    const int nf = g.F;
    const int b = std::min(B, nf);
    std::vector<int32_t> ns = g.n_per, es = g.e_per;
    std::sort(ns.begin(), ns.end());
    std::sort(es.begin(), es.end());
    int n_max = 0, e_max = 0;
    for (int i = nf - b; i < nf; ++i) {
        n_max += ns[i];
        e_max += es[i];
    }
    int sh = nf % B;
    if (sh == 0) sh = B;
    int e_min = 0;
    for (int i = 0; i < std::min(sh, nf); ++i) e_min += es[i];
    int D = std::max(1, (e_max - e_min + DUMMY_ROW - 1) / DUMMY_ROW + 1);
    LstShape s{};
    s.N = n_max + D;
    s.E = e_max + (e_max % 2);
    s.P = s.E / 2;
    s.G1 = B + 1;
    s.B = B;
    return s;
}

// Exact 27 LstMuonBlock entries for Loup-S (matching loupmd/fasttrain/muon.py::library_blocks)
std::vector<LstMuonBlock> build_muon_blocks(const LsmModel& m) {
    const int S = m.S, C = m.C, H = m.H, NB = m.NB, K3 = m.K3;
    std::vector<LstMuonBlock> out;
    auto add = [&](int p, int b, int r, int c, long off, long bs, long rs, long cs) {
        if (r > 64 && c > 64) {
            for (int r0 = 0; r0 < r; r0 += 64)
                for (int c0 = 0; c0 < c; c0 += 64)
                    out.push_back(LstMuonBlock{p, b, std::min(64, r - r0), std::min(64, c - c0),
                                               off + (long)r0 * rs + (long)c0 * cs, bs, rs, cs});
        } else {
            out.push_back(LstMuonBlock{p, b, r, c, off, bs, rs, cs});
        }
    };
    // Layer 0: linear_up (1), conv_tp_weights.layer0..3 (2..5), linear (6), skip_tp (7)
    add(1, 1, C, C, 0, C * C, C, 1);
    add(2, 1, NB, H, 0, NB * H, H, 1);
    add(3, 1, H, H, 0, H * H, H, 1);
    add(4, 1, H, H, 0, H * H, H, 1);
    add(5, 1, H, 4 * C, 0, H * 4 * C, 4 * C, 1);
    for (int l = 0; l < 4; ++l) add(6, 1, C, C, (long)l * C * C, C * C, C, 1);
    for (int l = 0; l < 4; ++l) add(7, S, C, C, (long)l * S * C * C, C, (long)S * C, 1);
    // Layer 1: linear_up (8), conv_tp_weights.layer0..3 (9..12), linear (13), skip_tp (14)
    add(8, 1, C, C, 0, C * C, C, 1);
    add(9, 1, NB, H, 0, NB * H, H, 1);
    add(10, 1, H, H, 0, H * H, H, 1);
    add(11, 1, H, H, 0, H * H, H, 1);
    add(12, 1, H, 4 * C, 0, H * 4 * C, 4 * C, 1);
    for (int l = 0; l < 4; ++l) add(13, 1, C, C, (long)l * C * C, C * C, C, 1);
    add(14, S, C, C, 0, C, (long)S * C, 1);
    // Products 0 & 1: weights_max (15, 19), linear (18, 22)
    add(15, S, K3, C, 0, (long)K3 * C, C, 1);
    add(18, 1, C, C, 0, C * C, C, 1);
    add(19, S, K3, C, 0, (long)K3 * C, C, 1);
    add(22, 1, C, C, 0, C * C, C, 1);
    return out;
}

// ============================================================================ Recipe config + lightweight JSON loader
struct WeightPhase {
    double from_epoch = 0.0;
    double energy = 1.0, forces = 100.0, stress = 30.0;
    double ramp_epochs = 0.0;
};

struct RecipeCfg {
    std::string name = "soap40";
    int epochs = 40;
    int batch_size = 1;
    int seed = 1;
    // LR schedule
    std::string lr_schedule = "piecewise";
    double lr_peak = 0.05, lr_final = 1e-4, lr_warmup = 0.0, lr_warmup_start = 0.0;
    std::vector<std::pair<double, double>> lr_points = {{0.0, 0.005}, {1.0, 0.05}, {36.0, 0.0015}, {40.0, 0.0001}};
    std::vector<WeightPhase> weights = {{0.0, 1.0, 100.0, 30.0, 0.0}, {36.0, 1000.0, 100.0, 10.0, 1.0}};
    double ema_decay = 0.99;
    double max_grad_norm = 1.0;
    double update_clip = 0.0;
    int eval_every = 0;  // 0 = final only
    // Optimizer
    bool use_muon = true;
    bool use_soap = true;
    bool cautious = true;
    bool amsgrad = true;   // mace-torch distill runs pass --amsgrad; recipe JSON may override
    double beta1 = 0.9, beta2 = 0.99, eps = 1e-8, weight_decay = 0.0;
    double muon_momentum = 0.95, muon_scale = 0.2, muon_lr_mult = 1.5, muon_wd = 0.0;
    int ns_steps = 8, ns_polish = 2;
    double soap_b1 = 0.995, soap_b2 = 0.995;
    int soap_every = 10, soap_max_dim = 64, soap_sweeps = 2;
    bool soap_block = true;
    // Sampling
    bool use_sampling = true;
    double smp_power = 2.0, smp_uniform = 0.5, smp_is_beta = 0.5, smp_smooth = 0.5;
    int smp_from_epoch = 1;

    double lr_at(double t) const {
        if (lr_warmup > 0.0 && t < lr_warmup) {
            double st = lr_peak * lr_warmup_start;
            return st + (lr_peak - st) * (t / lr_warmup);
        }
        if (lr_schedule == "piecewise" && !lr_points.empty()) {
            if (t <= lr_points.front().first) return lr_points.front().second;
            for (size_t i = 0; i + 1 < lr_points.size(); ++i) {
                if (t <= lr_points[i + 1].first) {
                    double e0 = lr_points[i].first, v0 = lr_points[i].second;
                    double e1 = lr_points[i + 1].first, v1 = lr_points[i + 1].second;
                    return v0 + (v1 - v0) * (t - e0) / std::max(e1 - e0, 1e-12);
                }
            }
            return lr_points.back().second;
        }
        double span = std::max((double)epochs - lr_warmup, 1e-12);
        double u = std::min(std::max((t - lr_warmup) / span, 0.0), 1.0);
        if (lr_schedule == "constant") return lr_peak;
        if (lr_schedule == "cosine") return lr_final + 0.5 * (lr_peak - lr_final) * (1.0 + std::cos(M_PI * u));
        if (lr_schedule == "linear") return lr_peak + (lr_final - lr_peak) * u;
        return lr_peak * std::pow(lr_final / lr_peak, u);
    }

    void weights_at(double t, double out[3]) const {
        out[0] = weights[0].energy;
        out[1] = weights[0].forces;
        out[2] = weights[0].stress;
        for (size_t i = 1; i < weights.size(); ++i) {
            const auto& ph = weights[i];
            const auto& prev = weights[i - 1];
            if (t < ph.from_epoch) break;
            double a = (ph.ramp_epochs <= 0.0) ? 1.0 : std::min((t - ph.from_epoch) / ph.ramp_epochs, 1.0);
            out[0] = prev.energy + (ph.energy - prev.energy) * a;
            out[1] = prev.forces + (ph.forces - prev.forces) * a;
            out[2] = prev.stress + (ph.stress - prev.stress) * a;
        }
    }
};

// Minimal recursive JSON value for recipe files
struct JVal {
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ } type = NUL;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<JVal> arr;
    std::vector<std::pair<std::string, JVal>> obj;
    bool has(const std::string& k) const {
        if (type != OBJ) return false;
        for (const auto& kv : obj)
            if (kv.first == k) return true;
        return false;
    }
    const JVal& at(const std::string& k) const {
        for (const auto& kv : obj)
            if (kv.first == k) return kv.second;
        std::fprintf(stderr, "Missing JSON key: %s\n", k.c_str());
        std::exit(1);
    }
    void set(const std::string& k, JVal&& v) {
        for (auto& kv : obj) {
            if (kv.first == k) { kv.second = std::move(v); return; }
        }
        obj.emplace_back(k, std::move(v));
    }
};

struct JParser {
    const std::string& s;
    size_t i = 0;
    void ws() { while (i < s.size() && std::isspace((unsigned char)s[i])) ++i; }
    std::string parse_str() {
        ++i;  // '"'
        std::string r;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) ++i;
            r.push_back(s[i++]);
        }
        if (i < s.size()) ++i;
        return r;
    }
    JVal parse() {
        ws();
        JVal v;
        if (i >= s.size()) return v;
        char c = s[i];
        if (c == '"') { v.type = JVal::STR; v.str = parse_str(); return v; }
        if (c == '{') {
            v.type = JVal::OBJ;
            ++i;
            ws();
            if (i < s.size() && s[i] == '}') { ++i; return v; }
            while (i < s.size()) {
                ws();
                std::string k = parse_str();
                ws();
                if (i < s.size() && s[i] == ':') ++i;
                v.set(k, parse());
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; break; }
            }
            return v;
        }
        if (c == '[') {
            v.type = JVal::ARR;
            ++i;
            ws();
            if (i < s.size() && s[i] == ']') { ++i; return v; }
            while (i < s.size()) {
                v.arr.push_back(parse());
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; break; }
            }
            return v;
        }
        if (s.compare(i, 4, "true") == 0) { v.type = JVal::BOOL; v.b = true; i += 4; return v; }
        if (s.compare(i, 5, "false") == 0) { v.type = JVal::BOOL; v.b = false; i += 5; return v; }
        if (s.compare(i, 4, "null") == 0) { v.type = JVal::NUL; i += 4; return v; }
        char* ep = nullptr;
        v.type = JVal::NUM;
        v.num = std::strtod(s.c_str() + i, &ep);
        i = (size_t)(ep - s.c_str());
        return v;
    }
};

void load_recipe_json(const std::string& path, RecipeCfg& rc) {
    std::ifstream fh(path);
    if (!fh) {
        std::fprintf(stderr, "Cannot open recipe JSON: %s\n", path.c_str());
        std::exit(1);
    }
    std::ostringstream ss;
    ss << fh.rdbuf();
    std::string txt = ss.str();
    JParser jp{txt, 0};
    JVal root = jp.parse();
    if (root.has("name")) rc.name = root.at("name").str;
    if (root.has("epochs")) rc.epochs = (int)root.at("epochs").num;
    if (root.has("batch_size") && root.at("batch_size").type == JVal::NUM) rc.batch_size = (int)root.at("batch_size").num;
    if (root.has("seed")) rc.seed = (int)root.at("seed").num;
    if (root.has("ema_decay") && root.at("ema_decay").type == JVal::NUM) rc.ema_decay = root.at("ema_decay").num;
    if (root.has("max_grad_norm") && root.at("max_grad_norm").type == JVal::NUM) rc.max_grad_norm = root.at("max_grad_norm").num;
    if (root.has("lr")) {
        const auto& lr = root.at("lr");
        if (lr.has("schedule")) rc.lr_schedule = lr.at("schedule").str;
        if (lr.has("peak")) rc.lr_peak = lr.at("peak").num;
        if (lr.has("final")) rc.lr_final = lr.at("final").num;
        if (lr.has("warmup_epochs")) rc.lr_warmup = lr.at("warmup_epochs").num;
        if (lr.has("warmup_start")) rc.lr_warmup_start = lr.at("warmup_start").num;
        if (lr.has("points")) {
            rc.lr_points.clear();
            for (const auto& pt : lr.at("points").arr)
                rc.lr_points.push_back({pt.arr[0].num, pt.arr[1].num});
            std::sort(rc.lr_points.begin(), rc.lr_points.end());
        }
    }
    if (root.has("weights")) {
        rc.weights.clear();
        for (const auto& ph : root.at("weights").arr) {
            WeightPhase wp;
            if (ph.has("from_epoch")) wp.from_epoch = ph.at("from_epoch").num;
            if (ph.has("energy")) wp.energy = ph.at("energy").num;
            if (ph.has("forces")) wp.forces = ph.at("forces").num;
            if (ph.has("stress")) wp.stress = ph.at("stress").num;
            if (ph.has("ramp_epochs")) wp.ramp_epochs = ph.at("ramp_epochs").num;
            rc.weights.push_back(wp);
        }
        std::sort(rc.weights.begin(), rc.weights.end(),
                  [](const WeightPhase& a, const WeightPhase& b) { return a.from_epoch < b.from_epoch; });
    }
    if (root.has("optimizer")) {
        const auto& oc = root.at("optimizer");
        rc.use_muon = oc.has("kind") && oc.at("kind").str == "muon";
        rc.use_soap = rc.use_muon && oc.has("matrix") && oc.at("matrix").str == "soap";
        rc.cautious = oc.has("cautious") && oc.at("cautious").str == "adam";
        if (oc.has("amsgrad")) rc.amsgrad = oc.at("amsgrad").b;
        if (oc.has("betas") && oc.at("betas").arr.size() == 2) {
            rc.beta1 = oc.at("betas").arr[0].num;
            rc.beta2 = oc.at("betas").arr[1].num;
        }
        if (oc.has("eps")) rc.eps = oc.at("eps").num;
        if (oc.has("weight_decay")) rc.weight_decay = oc.at("weight_decay").num;
        if (oc.has("update_clip")) rc.update_clip = oc.at("update_clip").num;
        if (oc.has("momentum")) rc.muon_momentum = oc.at("momentum").num;
        if (oc.has("scale")) rc.muon_scale = oc.at("scale").num;
        if (oc.has("muon_lr_mult")) rc.muon_lr_mult = oc.at("muon_lr_mult").num;
        if (oc.has("muon_wd")) rc.muon_wd = oc.at("muon_wd").num;
        if (oc.has("ns_steps")) rc.ns_steps = (int)oc.at("ns_steps").num;
        if (oc.has("ns_polish")) rc.ns_polish = (int)oc.at("ns_polish").num;
        if (oc.has("soap_betas") && oc.at("soap_betas").arr.size() == 2) {
            rc.soap_b1 = oc.at("soap_betas").arr[0].num;
            rc.soap_b2 = oc.at("soap_betas").arr[1].num;
        }
        if (oc.has("soap_every")) rc.soap_every = (int)oc.at("soap_every").num;
        if (oc.has("soap_max_dim")) rc.soap_max_dim = (int)oc.at("soap_max_dim").num;
        if (oc.has("soap_sweeps")) rc.soap_sweeps = (int)oc.at("soap_sweeps").num;
        if (oc.has("soap_block")) rc.soap_block = oc.at("soap_block").b;
    }
    if (root.has("sampling") && root.at("sampling").type == JVal::OBJ) {
        const auto& sm = root.at("sampling");
        rc.use_sampling = true;
        if (sm.has("power")) rc.smp_power = sm.at("power").num;
        if (sm.has("uniform")) rc.smp_uniform = sm.at("uniform").num;
        if (sm.has("is_beta")) rc.smp_is_beta = sm.at("is_beta").num;
        if (sm.has("smooth")) rc.smp_smooth = sm.at("smooth").num;
        if (sm.has("from_epoch")) rc.smp_from_epoch = (int)sm.at("from_epoch").num;
    } else if (root.has("sampling") && root.at("sampling").type == JVal::NUL) {
        rc.use_sampling = false;
    }
}

// ============================================================================ Evaluation helper
struct EvalMetrics {
    double rmse_e_mev = 0.0;
    double rmse_f_mev = 0.0;
    double rmse_s_mev = 0.0;
    double rmse_v_mev = 0.0;
};

template <typename T>
EvalMetrics run_eval_t(LstEngine* ev_h, const LstShape& sh, const GpuFrameSet& g,
                       int f_beg, int f_end,
                       void* d_e, void* d_f, void* d_s, cudaStream_t st) {
    const int B = sh.B;
    const int F_ev = f_end - f_beg;
    int N_ev = 0;
    for (int idx = f_beg; idx < f_end; ++idx) N_ev += g.n_per[idx];
    T *d_e_all = nullptr, *d_f_all = nullptr, *d_s_all = nullptr;
    CK_CUDA(cudaMalloc(&d_e_all, (size_t)std::max(1, F_ev) * sizeof(T)));
    CK_CUDA(cudaMalloc(&d_f_all, (size_t)std::max(1, N_ev) * 3 * sizeof(T)));
    CK_CUDA(cudaMalloc(&d_s_all, (size_t)std::max(1, F_ev) * 9 * sizeof(T)));

    std::vector<int32_t> fids(B);
    int n_cur = 0;
    for (int i0 = f_beg; i0 < f_end; i0 += B) {
        const int k = std::min(B, f_end - i0);
        int nr = 0;
        for (int j = 0; j < B; ++j) {
            fids[j] = (j < k) ? (i0 + j) : (f_end - 1);
            if (j < k) nr += g.n_per[i0 + j];
        }
        CK_LST(ev_h, lst_evaluate(ev_h, fids.data(), B, d_e, d_f, d_s, st));
        CK_CUDA(cudaMemcpyAsync(d_e_all + (i0 - f_beg), d_e, (size_t)k * sizeof(T), cudaMemcpyDeviceToDevice, st));
        CK_CUDA(cudaMemcpyAsync(d_f_all + (size_t)n_cur * 3, d_f, (size_t)nr * 3 * sizeof(T), cudaMemcpyDeviceToDevice, st));
        CK_CUDA(cudaMemcpyAsync(d_s_all + (size_t)(i0 - f_beg) * 9, d_s, (size_t)k * 9 * sizeof(T), cudaMemcpyDeviceToDevice, st));
        n_cur += nr;
    }

    std::vector<T> eh(F_ev), fh((size_t)N_ev * 3), sh_buf((size_t)F_ev * 9);
    CK_CUDA(cudaMemcpyAsync(eh.data(), d_e_all, (size_t)F_ev * sizeof(T), cudaMemcpyDeviceToHost, st));
    CK_CUDA(cudaMemcpyAsync(fh.data(), d_f_all, (size_t)N_ev * 3 * sizeof(T), cudaMemcpyDeviceToHost, st));
    CK_CUDA(cudaMemcpyAsync(sh_buf.data(), d_s_all, (size_t)F_ev * 9 * sizeof(T), cudaMemcpyDeviceToHost, st));
    CK_CUDA(cudaStreamSynchronize(st));
    cudaFree(d_e_all); cudaFree(d_f_all); cudaFree(d_s_all);

    double se2 = 0.0, sf2 = 0.0, ss2 = 0.0, sv2 = 0.0;
    int64_t ne = 0, nf = 0, ns = 0;
    int r0 = 0;
    for (int j = 0; j < F_ev; ++j) {
        const int idx = f_beg + j;
        const int na = g.n_per[idx];
        const double vol = g.vols_h[idx];
        const double de = ((double)eh[j] - g.e_ref_h[idx]) / (double)na;
        se2 += de * de;
        ++ne;
        const int n0 = g.node_off_h[idx];
        for (int a = 0; a < na * 3; ++a) {
            const double df = (double)fh[(size_t)r0 * 3 + a] - g.f_ref_sorted_h[(size_t)n0 * 3 + a];
            sf2 += df * df;
        }
        nf += (int64_t)na * 3;
        r0 += na;
        const double v_per_a = vol / (double)na;
        for (int c = 0; c < 9; ++c) {
            const double ds = (double)sh_buf[(size_t)j * 9 + c] - g.s_ref_h[(size_t)idx * 9 + c];
            ss2 += ds * ds;
            const double dv = ds * v_per_a;
            sv2 += dv * dv;
        }
        ns += 9;
    }
    EvalMetrics em;
    em.rmse_e_mev = std::sqrt(se2 / std::max<int64_t>(1, ne)) * 1e3;
    em.rmse_f_mev = std::sqrt(sf2 / std::max<int64_t>(1, nf)) * 1e3;
    em.rmse_s_mev = std::sqrt(ss2 / std::max<int64_t>(1, ns)) * 1e3;
    em.rmse_v_mev = std::sqrt(sv2 / std::max<int64_t>(1, ns)) * 1e3;
    return em;
}

// ============================================================================ Importance sampling (_draw)
void draw_epoch_batches(std::mt19937_64& rng, int n, int bs, int spe,
                        const std::vector<double>& score, bool have_scores,
                        const RecipeCfg& rc,
                        std::vector<int32_t>& frame_ids_out,
                        std::vector<double>& fac_out) {
    const int m = spe * bs;
    frame_ids_out.assign(m, -1);
    fac_out.assign(spe, 1.0);
    if (!rc.use_sampling || !have_scores) {
        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int i = 0; i < m; ++i) frame_ids_out[i] = perm[i];
        return;
    }
    const double u = rc.smp_uniform;
    std::vector<double> w(n), p(n);
    double wsum = 0.0;
    for (int i = 0; i < n; ++i) {
        w[i] = std::pow(std::max(score[i], 1e-12), rc.smp_power);
        wsum += w[i];
    }
    for (int i = 0; i < n; ++i) p[i] = u / (double)n + (1.0 - u) * (w[i] / wsum);

    const int k = (int)std::llround(u * (double)m);
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int> idx(m);
    for (int i = 0; i < k; ++i) idx[i] = perm[i];
    std::discrete_distribution<int> dist(w.begin(), w.end());
    for (int i = k; i < m; ++i) idx[i] = dist(rng);
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int i = 0; i < m; ++i) frame_ids_out[i] = idx[i];

    if (rc.smp_is_beta > 0.0) {
        double fmean = 0.0;
        for (int s = 0; s < spe; ++s) {
            double acc = 0.0;
            for (int j = 0; j < bs; ++j) acc += std::pow((1.0 / (double)n) / p[idx[s * bs + j]], rc.smp_is_beta);
            fac_out[s] = acc / (double)bs;
            fmean += fac_out[s];
        }
        fmean /= (double)spe;
        for (int s = 0; s < spe; ++s) fac_out[s] /= fmean;
    }
}

// ============================================================================ Exact 80-param Mode-1 VarPro
template <typename T>
double run_varpro_t(LstEngine* h, const LstShape& sh, const GpuFrameSet& g, int F_tr,
                    const LsmModel& m, T* d_params_all, T* d_ema_all, bool has_ema,
                    T* d_ev_e, T* d_ev_f, T* d_ev_s, double ridge_rel, cudaStream_t st) {
    auto tv0 = Clock::now();
    const int B = sh.B;
    const int C = m.C, hid = m.hid, D = C + hid;
    const T* d_f0_ptr = static_cast<const T*>(lst_stage_ptr(h, "f0"));
    const T* d_tt_ptr = static_cast<const T*>(lst_stage_ptr(h, "tt"));
    T *d_f0_tmp = nullptr, *d_tt_tmp = nullptr;
    if (!d_f0_ptr || !d_tt_ptr) {
        CK_CUDA(cudaMalloc(&d_f0_tmp, (size_t)sh.N * C * sizeof(T)));
        CK_CUDA(cudaMalloc(&d_tt_tmp, (size_t)sh.N * hid * sizeof(T)));
    }
    int32_t* d_bn_all = nullptr;
    double *d_e_all = nullptr, *d_s0_all = nullptr, *d_s1_all = nullptr;
    CK_CUDA(cudaMalloc(&d_bn_all, (size_t)F_tr * sizeof(int32_t)));
    CK_CUDA(cudaMalloc(&d_e_all, (size_t)F_tr * sizeof(double)));
    CK_CUDA(cudaMalloc(&d_s0_all, (size_t)F_tr * C * sizeof(double)));
    CK_CUDA(cudaMalloc(&d_s1_all, (size_t)F_tr * hid * sizeof(double)));
    CK_CUDA(cudaMemcpyAsync(d_bn_all, g.n_per.data(), (size_t)F_tr * sizeof(int32_t), cudaMemcpyHostToDevice, st));

    std::vector<int32_t> fids(B);
    std::vector<double> e_pred(F_tr, 0.0);
    std::vector<double> S0((size_t)F_tr * C, 0.0), S1((size_t)F_tr * hid, 0.0);

    // Wave 33: Energy-only forward + frozen-weights eval cache + direct lst_stage_ptr (zero 600 D2D copies!) + FP32 sums!
    CK_LST(h, lst_set_eval_cache(h, 1));
    for (int i0 = 0; i0 < F_tr; i0 += B) {
        const int k = std::min(B, F_tr - i0);
        for (int j = 0; j < B; ++j) fids[j] = (j < k) ? (i0 + j) : (F_tr - 1);
        CK_LST(h, lst_evaluate(h, fids.data(), B, d_ev_e, nullptr, nullptr, st));
        const T *pf0 = d_f0_ptr, *ptt = d_tt_ptr;
        if (!pf0 || !ptt) {
            CK_LST(h, lst_stage_copy(h, "f0", d_f0_tmp, st));
            CK_LST(h, lst_stage_copy(h, "tt", d_tt_tmp, st));
            pf0 = d_f0_tmp; ptt = d_tt_tmp;
        }
        k_varpro_frame_sums<T><<<k, 256, 0, st>>>(
            i0, k, C, hid, d_bn_all + i0, d_ev_e, pf0, ptt, d_e_all, d_s0_all, d_s1_all);
    }
    CK_LST(h, lst_set_eval_cache(h, 0));
    CK_CUDA(cudaMemcpyAsync(e_pred.data(), d_e_all, (size_t)F_tr * sizeof(double), cudaMemcpyDeviceToHost, st));
    CK_CUDA(cudaMemcpyAsync(S0.data(), d_s0_all, (size_t)F_tr * C * sizeof(double), cudaMemcpyDeviceToHost, st));
    CK_CUDA(cudaMemcpyAsync(S1.data(), d_s1_all, (size_t)F_tr * hid * sizeof(double), cudaMemcpyDeviceToHost, st));
    CK_CUDA(cudaStreamSynchronize(st));

    std::vector<size_t> poff(LST_N_PARAMS, 0);
    for (int i = 1; i < LST_N_PARAMS; ++i) poff[i] = poff[i - 1] + (size_t)m.param_numel[i - 1];
    std::vector<T> wp0_h(C * C), wro0_h(C), wro2_h(hid), core_e_h(F_tr);
    CK_CUDA(cudaMemcpy(wp0_h.data(), d_params_all + poff[18], (size_t)C * C * sizeof(T), cudaMemcpyDeviceToHost));
    CK_CUDA(cudaMemcpy(wro0_h.data(), d_params_all + poff[23], (size_t)C * sizeof(T), cudaMemcpyDeviceToHost));
    CK_CUDA(cudaMemcpy(wro2_h.data(), d_params_all + poff[25], (size_t)hid * sizeof(T), cudaMemcpyDeviceToHost));
    CK_CUDA(cudaMemcpy(core_e_h.data(), g.d_core_e, (size_t)F_tr * sizeof(T), cudaMemcpyDeviceToHost));

    // Exact linear design matrix A [F_tr, C + hid] for w = [raw[23] (readouts.0), raw[25] (readouts.1.linear_2)]
    std::vector<double> A((size_t)F_tr * D, 0.0);
    std::vector<double> w_cur(D, 0.0);
    for (int c = 0; c < C; ++c) w_cur[c] = (double)wro0_h[c];
    for (int q = 0; q < hid; ++q) w_cur[C + q] = (double)wro2_h[q];

    const double fac0 = m.scale / (double)C;
    const double fac1 = m.scale / std::sqrt((double)hid);
    double chk_max = 0.0, mse_before = 0.0;
    std::vector<double> G((size_t)D * D, 0.0), rhs(D, 0.0);
    for (int f = 0; f < F_tr; ++f) {
        double* af = A.data() + (size_t)f * D;
        const double* s0f = S0.data() + (size_t)f * C;
        const double* s1f = S1.data() + (size_t)f * hid;
        for (int c = 0; c < C; ++c) {
            double acc = 0.0;
            for (int k = 0; k < C; ++k) acc += s0f[k] * (double)wp0_h[(size_t)k * C + c];
            af[c] = fac0 * acc;
        }
        for (int q = 0; q < hid; ++q) af[C + q] = fac1 * s1f[q];
        double e_chk = (double)core_e_h[f];
        for (int a = 0; a < D; ++a) e_chk += af[a] * w_cur[a];
        chk_max = std::max(chk_max, std::fabs(e_chk - e_pred[f]));

        const double inv_na = 1.0 / (double)g.n_per[f];
        const double r_at = (g.e_ref_h[f] - e_pred[f]) * inv_na;
        mse_before += r_at * r_at;
        for (int a = 0; a < D; ++a) {
            const double aa = af[a] * inv_na;
            rhs[a] += aa * r_at;
            for (int b = 0; b <= a; ++b) G[(size_t)a * D + b] += aa * (af[b] * inv_na);
        }
    }
    for (int a = 0; a < D; ++a)
        for (int b = a + 1; b < D; ++b) G[(size_t)a * D + b] = G[(size_t)b * D + a];

    double tr = 0.0;
    for (int a = 0; a < D; ++a) tr += G[(size_t)a * D + a];
    const double lam = std::max(1e-18, ridge_rel * tr / (double)D);
    for (int a = 0; a < D; ++a) G[(size_t)a * D + a] += lam;

    std::vector<double> dw;
    bool ok = solve_spd_cholesky(D, G, rhs, dw);
    double dw_norm = 0.0, mse_after = mse_before;
    if (ok) {
        for (int a = 0; a < D; ++a) dw_norm += dw[a] * dw[a];
        dw_norm = std::sqrt(dw_norm);
        mse_after = 0.0;
        for (int f = 0; f < F_tr; ++f) {
            const double* af = A.data() + (size_t)f * D;
            double de_add = 0.0;
            for (int a = 0; a < D; ++a) de_add += af[a] * dw[a];
            const double r_at = (g.e_ref_h[f] - (e_pred[f] + de_add)) / (double)g.n_per[f];
            mse_after += r_at * r_at;
        }
        for (int c = 0; c < C; ++c) wro0_h[c] = (T)((double)wro0_h[c] + dw[c]);
        for (int q = 0; q < hid; ++q) wro2_h[q] = (T)((double)wro2_h[q] + dw[C + q]);
        CK_CUDA(cudaMemcpyAsync(d_params_all + poff[23], wro0_h.data(), (size_t)C * sizeof(T), cudaMemcpyHostToDevice, st));
        CK_CUDA(cudaMemcpyAsync(d_params_all + poff[25], wro2_h.data(), (size_t)hid * sizeof(T), cudaMemcpyHostToDevice, st));
        if (has_ema) {
            CK_CUDA(cudaMemcpyAsync(d_ema_all + poff[23], wro0_h.data(), (size_t)C * sizeof(T), cudaMemcpyHostToDevice, st));
            CK_CUDA(cudaMemcpyAsync(d_ema_all + poff[25], wro2_h.data(), (size_t)hid * sizeof(T), cudaMemcpyHostToDevice, st));
        }
        CK_CUDA(cudaStreamSynchronize(st));
    }

    if (d_f0_tmp) cudaFree(d_f0_tmp); if (d_tt_tmp) cudaFree(d_tt_tmp);
    cudaFree(d_bn_all); cudaFree(d_e_all); cudaFree(d_s0_all); cudaFree(d_s1_all);
    const double dt_s = elapsed_s(tv0);
    std::printf("  VARPRO (ridge=%.1e, %.1fms): chk=%.2e eV | |dw|=%.4f | train_E: %.3f -> %.3f meV/at%s\n",
                ridge_rel, dt_s * 1e3, chk_max, dw_norm,
                std::sqrt(mse_before / std::max(1, F_tr)) * 1e3,
                std::sqrt(mse_after / std::max(1, F_tr)) * 1e3,
                ok ? "" : " [SOLVE_FAILED]");
    return dt_s;
}

struct CliExtras {
    bool use_varpro = false;
    double varpro_ridge = 1e-2;
    int ramp_ep = -1;
    int ramp_b2 = -1;
    double ramp_lr_mult = 1.0;
    double ramp_ema_decay = -1.0;
    double w_ei = 0.0;
    double ei_decay_start = 0.6;
    double ei_decay_end = 0.9;
};

// ============================================================================ Main training driver
template <typename T>
int run_training_t(LsmModel& m, const RecipeCfg& rc, const CliExtras& ex,
                   const std::string& train_path, const std::string& valid_path,
                   const std::string& test_path, const std::string& out_path,
                   bool center_e0, bool auto_stats, bool dump_prep_check,
                   Clock::time_point t_proc0) {
    // 1. Wave 31: overlap multi-threaded extXYZ parsing + E0 centering + auto-stats with CUDA context init!
    std::vector<RawFrame> raw_tr, raw_va, raw_te;
    std::vector<double> physical_e0 = m.E0;
    bool did_center = false;
    double t_parse_bg = 0.0;
    std::thread parse_thr([&]() {
        auto tp0 = Clock::now();
        raw_tr = load_extxyz(train_path, m);
        if (!valid_path.empty()) raw_va = load_extxyz(valid_path, m);
        if (!test_path.empty()) raw_te = load_extxyz(test_path, m);
        if (ex.w_ei > 0.0) {
            for (size_t f = 0; f < raw_tr.size(); ++f) {
                if (!raw_tr[f].has_node_energy) {
                    std::fprintf(stderr, "ERROR: --w-ei %.3g requires REF_node_energy on all training frames (missing on frame %zu)\n",
                                 ex.w_ei, f);
                    std::exit(1);
                }
            }
        }
        if (center_e0 && m.dtype == 0) {
            physical_e0 = fit_min_norm_e0(raw_tr, m.S);
            auto sub_e0 = [&](std::vector<RawFrame>& vec) {
                for (auto& rf : vec) {
                    double e0sum = 0.0;
                    for (int a = 0; a < rf.n; ++a) {
                        const double e0a = physical_e0[rf.sp[a]];
                        e0sum += e0a;
                        if (rf.has_node_energy) rf.node_energy[a] -= e0a;
                    }
                    rf.energy -= e0sum;
                }
            };
            sub_e0(raw_tr);
            sub_e0(raw_va);
            sub_e0(raw_te);
            for (int s = 0; s < m.S; ++s) m.E0[s] = 0.0;
            did_center = true;
        }
        if (auto_stats) {
            double mean_e = 0.0, f2_sum = 0.0;
            int64_t f_cnt = 0;
            for (const auto& rf : raw_tr) {
                double e0s = 0.0;
                for (int sp : rf.sp) e0s += m.E0[sp];
                mean_e += (rf.energy - e0s) / (double)rf.n;
                for (double fv : rf.force) f2_sum += fv * fv;
                f_cnt += (int64_t)rf.force.size();
            }
            m.shift = mean_e / (double)raw_tr.size();
            m.scale = std::sqrt(f2_sum / (double)std::max<int64_t>(1, f_cnt));
        }
        t_parse_bg = elapsed_s(tp0);
    });

    cudaStream_t st = nullptr;
    CK_CUDA(cudaStreamCreate(&st));
    auto t0 = Clock::now();
    parse_thr.join();
    const double t_parse = elapsed_s(t0);  // remaining wait after CUDA init (t_parse_bg overlapped!)

    // 2. Single concatenated GPU frame set [train | valid | test] -> ONE LstEngine handle!
    auto t1 = Clock::now();
    const int F_tr = (int)raw_tr.size();
    const int F_va = (int)raw_va.size();
    const int F_te = (int)raw_te.size();
    std::vector<RawFrame> raw_all;
    raw_all.reserve((size_t)(F_tr + F_va + F_te));
    for (auto& rf : raw_tr) raw_all.push_back(std::move(rf));
    for (auto& rf : raw_va) raw_all.push_back(std::move(rf));
    for (auto& rf : raw_te) raw_all.push_back(std::move(rf));

    GpuFrameSet g_all;
    build_gpu_frames_t<T>(raw_all, m, g_all, st);
    if (auto_stats) {
        int64_t esum = 0, nsum = 0;
        for (int f = 0; f < F_tr; ++f) {
            esum += g_all.e_per[f];
            nsum += g_all.n_per[f];
        }
        m.avg0 = m.avg1 = nsum > 0 ? (double)esum / (double)nsum : 1.0;
    }
    const double t_prep = elapsed_s(t1);

    if (dump_prep_check) {
        // Print exact frame 0 diagnostics for bit-level parity check against Python data.py
        std::vector<T> ce(F_tr), cf(g_all.n_per[0] * 3), cs(9);
        CK_CUDA(cudaMemcpy(ce.data(), g_all.d_core_e, F_tr * sizeof(T), cudaMemcpyDeviceToHost));
        CK_CUDA(cudaMemcpy(cf.data(), g_all.d_core_f, g_all.n_per[0] * 3 * sizeof(T), cudaMemcpyDeviceToHost));
        CK_CUDA(cudaMemcpy(cs.data(), g_all.d_core_s, 9 * sizeof(T), cudaMemcpyDeviceToHost));
        double cf_norm = 0.0, cs_norm = 0.0;
        for (T v : cf) cf_norm += (double)v * (double)v;
        for (T v : cs) cs_norm += (double)v * (double)v;
        int64_t tr_e = 0, tr_n = 0;
        for (int f = 0; f < F_tr; ++f) { tr_e += g_all.e_per[f]; tr_n += g_all.n_per[f]; }
        std::printf("PREP_CHECK F=%d N_tot=%lld E_tot=%lld P_tot=%lld avg_nb=%.6f | f0: E=%d core_e=%.10f |core_f|=%.10f |core_s|=%.10f\n",
                    F_tr, (long long)tr_n, (long long)tr_e, (long long)(tr_e / 2), (double)tr_e / (double)tr_n,
                    g_all.e_per[0], (double)ce[0], std::sqrt(cf_norm), std::sqrt(cs_norm));
    }

    // 3. Create ONE LstEngine (handles train + valid + test eval without extra cuBLAS init!)
    auto t2 = Clock::now();
    LstModelDesc md{};
    md.dtype = m.dtype;
    md.S = m.S;
    md.C = m.C;
    md.H = m.H;
    md.NB = m.NB;
    md.hid = m.hid;
    md.K = m.K;
    md.K3 = m.K3;
    md.K2 = m.K2;
    md.p_cut = m.p_cut;
    md.r_max = m.r_max;
    md.prefactor = m.prefactor;
    md.act = m.act;
    md.scale = m.scale;
    md.avg0 = m.avg0;
    md.avg1 = m.avg1;
    md.bessel_w = m.bessel_w.data();

    const size_t tot_p = m.total_params();
    T *d_params_all = nullptr, *d_ema_all = nullptr, *d_tmp_p = nullptr;
    T *d_m1_all = nullptr, *d_m2_all = nullptr, *d_vmax_all = nullptr;
    float* d_steps_all = nullptr;
    CK_CUDA(cudaMalloc(&d_params_all, tot_p * sizeof(T)));
    CK_CUDA(cudaMalloc(&d_ema_all, tot_p * sizeof(T)));
    CK_CUDA(cudaMalloc(&d_tmp_p, tot_p * sizeof(T)));
    CK_CUDA(cudaMalloc(&d_m1_all, tot_p * sizeof(T)));
    CK_CUDA(cudaMalloc(&d_m2_all, tot_p * sizeof(T)));
    if (rc.amsgrad) CK_CUDA(cudaMalloc(&d_vmax_all, tot_p * sizeof(T)));
    CK_CUDA(cudaMalloc(&d_steps_all, LST_N_PARAMS * sizeof(float)));

    const void* h_p_src = (m.dtype == 0) ? (const void*)m.params_f32.data() : (const void*)m.params_f64.data();
    CK_CUDA(cudaMemcpyAsync(d_params_all, h_p_src, tot_p * sizeof(T), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemcpyAsync(d_ema_all, h_p_src, tot_p * sizeof(T), cudaMemcpyHostToDevice, st));
    CK_CUDA(cudaMemsetAsync(d_m1_all, 0, tot_p * sizeof(T), st));
    CK_CUDA(cudaMemsetAsync(d_m2_all, 0, tot_p * sizeof(T), st));
    if (d_vmax_all) CK_CUDA(cudaMemsetAsync(d_vmax_all, 0, tot_p * sizeof(T), st));
    CK_CUDA(cudaMemsetAsync(d_steps_all, 0, LST_N_PARAMS * sizeof(float), st));

    void* p_ptrs[LST_N_PARAMS];
    void* ema_ptrs[LST_N_PARAMS];
    LstOptim opt{};
    size_t off = 0;
    for (int i = 0; i < LST_N_PARAMS; ++i) {
        p_ptrs[i] = d_params_all + off;
        ema_ptrs[i] = d_ema_all + off;
        opt.exp_avg[i] = d_m1_all + off;
        opt.exp_avg_sq[i] = d_m2_all + off;
        opt.max_exp_avg_sq[i] = d_vmax_all ? (void*)(d_vmax_all + off) : nullptr;
        opt.step[i] = d_steps_all + i;
        opt.ema[i] = (rc.ema_decay > 0.0) ? ema_ptrs[i] : nullptr;
        opt.group[i] = 0;
        off += (size_t)m.param_numel[i];
    }
    opt.n_groups = 1;
    opt.beta1 = rc.beta1;
    opt.beta2 = rc.beta2;
    opt.eps = rc.eps;
    opt.weight_decay[0] = rc.weight_decay;
    opt.max_grad_norm = rc.max_grad_norm;
    opt.update_clip = rc.update_clip;

    const auto muon_blocks = rc.use_muon ? build_muon_blocks(m) : std::vector<LstMuonBlock>{};
    T *d_ev_e = nullptr, *d_ev_f = nullptr, *d_ev_s = nullptr;
    auto alloc_eval_bufs = [&](const LstShape& s) {
        if (d_ev_e) { cudaFree(d_ev_e); cudaFree(d_ev_f); cudaFree(d_ev_s); d_ev_e = d_ev_f = d_ev_s = nullptr; }
        if (F_va > 0 || F_te > 0 || ex.use_varpro) {
            CK_CUDA(cudaMalloc(&d_ev_e, s.G1 * sizeof(T)));
            CK_CUDA(cudaMalloc(&d_ev_f, s.N * 3 * sizeof(T)));
            CK_CUDA(cudaMalloc(&d_ev_s, s.G1 * 9 * sizeof(T)));
        }
    };
    auto create_engine = [&](int batch_sz, LstShape& sh_out) -> LstEngine* {
        sh_out = compute_shape(g_all, batch_sz);
        LstEngine* eng = lst_create(&md, &sh_out, 0);
        if (!eng) {
            std::fprintf(stderr, "lst_create failed: %s\n", lst_last_error(nullptr));
            std::exit(1);
        }
        CK_LST(eng, lst_set_params(eng, p_ptrs));
        CK_LST(eng, lst_set_frames(eng, &g_all.desc));
        CK_LST(eng, lst_set_node_order(eng, g_all.d_node_order));
        if (ex.w_ei > 0.0 && g_all.d_ei_ref) {
            CK_LST(eng, lst_set_node_energy(eng, g_all.d_ei_ref, ex.w_ei));
        }
        CK_LST(eng, lst_set_optim(eng, &opt));
        if (rc.use_muon) {
            CK_LST(eng, lst_set_muon(eng, muon_blocks.data(), (int)muon_blocks.size(), rc.muon_momentum, 1,
                                     rc.muon_scale * rc.muon_lr_mult));
            CK_LST(eng, lst_set_muon_ns(eng, rc.ns_steps, rc.ns_polish));
            if (rc.cautious) CK_LST(eng, lst_set_cautious(eng, 1));
            if (rc.muon_wd > 0.0) CK_LST(eng, lst_set_muon_wd(eng, rc.muon_wd * rc.muon_lr_mult));
            if (rc.use_soap) {
                CK_LST(eng, lst_set_soap(eng, rc.soap_b1, rc.soap_b2, rc.eps, rc.soap_every, rc.soap_max_dim,
                                         rc.muon_lr_mult));
            }
        }
        alloc_eval_bufs(sh_out);
        return eng;
    };

    LstShape sh_tr{};
    LstEngine* h = create_engine(rc.batch_size, sh_tr);
    // Helper: evaluate EMA weights in-place via 2 us D2D swap (never drops captured training CUDA graph!)
    auto eval_range_ema = [&](int fb, int fe) -> EvalMetrics {
        if (rc.ema_decay > 0.0) {
            CK_CUDA(cudaMemcpyAsync(d_tmp_p, d_params_all, tot_p * sizeof(T), cudaMemcpyDeviceToDevice, st));
            CK_CUDA(cudaMemcpyAsync(d_params_all, d_ema_all, tot_p * sizeof(T), cudaMemcpyDeviceToDevice, st));
        }
        auto em = run_eval_t<T>(h, sh_tr, g_all, fb, fe, d_ev_e, d_ev_f, d_ev_s, st);
        if (rc.ema_decay > 0.0) {
            CK_CUDA(cudaMemcpyAsync(d_params_all, d_tmp_p, tot_p * sizeof(T), cudaMemcpyDeviceToDevice, st));
        }
        return em;
    };

    CK_CUDA(cudaStreamSynchronize(st));
    const double t_engine = elapsed_s(t2);

    if (dump_prep_check && rc.batch_size == 1 && F_tr >= 5) {
        int32_t f5[5] = {0, 1, 2, 3, 4};
        double lr5[5] = {0.005, 0.005, 0.005, 0.005, 0.005};
        double w5[15];
        double ed5[5];
        for (int i = 0; i < 5; ++i) {
            w5[i * 3 + 0] = 1.0; w5[i * 3 + 1] = 100.0; w5[i * 3 + 2] = 30.0;
            ed5[i] = std::min(rc.ema_decay, (2.0 + i) / (11.0 + i));
        }
        LstBatches bt5{5, f5};
        double l5[5];
        CK_LST(h, lst_begin_epoch_steps(h, &bt5, lr5, w5, ed5, st));
        CK_LST(h, lst_run_steps(h, 5, st));
        CK_LST(h, lst_epoch_losses(h, l5, st));
        std::printf("PARITY5 losses [0..4]: %.8f %.8f %.8f %.8f %.8f\n", l5[0], l5[1], l5[2], l5[3], l5[4]);
        // Cleanup and return early
        if (d_ev_e) { cudaFree(d_ev_e); cudaFree(d_ev_f); cudaFree(d_ev_s); }
        lst_destroy(h);
        cudaFree(d_params_all); cudaFree(d_ema_all); cudaFree(d_tmp_p);
        cudaFree(d_m1_all); cudaFree(d_m2_all);
        if (d_vmax_all) cudaFree(d_vmax_all);
        cudaFree(d_steps_all);
        cudaStreamDestroy(st);
        return 0;
    }

    // 4. Epoch training loop (supports optional --batch-ramp ep_switch:b2)
    const int N_tr = F_tr;
    int bs = rc.batch_size;
    int spe = N_tr / bs;
    double cur_lr_mult = 1.0;
    double cur_ema_decay = rc.ema_decay;
    std::mt19937_64 rng((uint64_t)rc.seed);
    std::vector<double> score(N_tr, -1.0);
    bool have_scores = false;
    std::vector<int32_t> frame_ids(spe * bs);
    std::vector<double> fac(spe), lr_steps(spe), w_steps(spe * 3), wei_steps(spe), ema_dec(spe), losses(spe);
    int64_t ema_updates = 0;
    int64_t total_steps = 0;
    // Wave 38 (LST_LAZY_EMA=1): when eval_every == 0, EMA weights from early steps decay by d^K < 1.12e-7 (< 1 FP32 ULP)
    // for K = ceil(16 / -ln(d)). Skip q.ema R+W (ema_dec = 1.0 -> omd = 0) until step total_steps_all - K, then write-only
    // copy q.ema = q.p (ema_dec = 0.0 -> omd = 1) on the transition step, then normal EMA over the final K steps!
    int64_t total_steps_all = 0;
    {
        int bs_sim = rc.batch_size;
        for (int ep = 0; ep < rc.epochs; ++ep) {
            if (ex.ramp_ep > 0 && ep == ex.ramp_ep && ex.ramp_b2 > 0 && ex.ramp_b2 != bs_sim) bs_sim = ex.ramp_b2;
            total_steps_all += (int64_t)(N_tr / bs_sim);
        }
    }
    const char* elema = std::getenv("LST_LAZY_EMA");
    const bool lazy_ema_env = !(elema && elema[0] == '0');
    const double max_ema_d = (ex.ramp_ep > 0 && ex.ramp_ema_decay > 0.0) ? std::max(rc.ema_decay, ex.ramp_ema_decay) : rc.ema_decay;
    int64_t lazy_ema_copy_step = 0;
    if (lazy_ema_env && rc.eval_every == 0 && max_ema_d > 0.0 && max_ema_d < 1.0) {
        const int64_t ema_tail = (int64_t)std::ceil(16.0 / -std::log(max_ema_d));
        if (total_steps_all > ema_tail + 1) lazy_ema_copy_step = total_steps_all - ema_tail;
    }

    auto w_ei_at = [&](double t) -> double {
        if (ex.w_ei <= 0.0) return 0.0;
        const double u = t / std::max(1.0, (double)rc.epochs);
        if (u <= ex.ei_decay_start) return ex.w_ei;
        if (u >= ex.ei_decay_end) return 0.0;
        const double span = std::max(1e-12, ex.ei_decay_end - ex.ei_decay_start);
        return ex.w_ei * (1.0 - (u - ex.ei_decay_start) / span);
    };

    auto t_tr0 = Clock::now();
    double train_gpu_s = 0.0, eval_s = 0.0, ramp_s = 0.0, varpro_s = 0.0;

    for (int ep = 0; ep < rc.epochs; ++ep) {
        if (ex.ramp_ep > 0 && ep == ex.ramp_ep && ex.ramp_b2 > 0 && ex.ramp_b2 != bs) {
            auto tr0 = Clock::now();
            CK_CUDA(cudaStreamSynchronize(st));
            lst_destroy(h);
            bs = ex.ramp_b2;
            spe = N_tr / bs;
            h = create_engine(bs, sh_tr);
            CK_CUDA(cudaStreamSynchronize(st));
            frame_ids.resize(spe * bs);
            fac.resize(spe);
            lr_steps.resize(spe);
            w_steps.resize(spe * 3);
            wei_steps.resize(spe);
            ema_dec.resize(spe);
            losses.resize(spe);
            cur_lr_mult = ex.ramp_lr_mult;
            if (ex.ramp_ema_decay >= 0.0) cur_ema_decay = ex.ramp_ema_decay;
            ramp_s += elapsed_s(tr0);
        }
        draw_epoch_batches(rng, N_tr, bs, spe, score, have_scores && ep >= rc.smp_from_epoch, rc, frame_ids, fac);
        for (int i = 0; i < spe; ++i) {
            const double t = (double)ep + (double)i / (double)spe;
            lr_steps[i] = rc.lr_at(t) * cur_lr_mult;
            double ws[3];
            rc.weights_at(t, ws);
            const double f_scale = fac[i];
            w_steps[i * 3 + 0] = ws[0] * f_scale;
            w_steps[i * 3 + 1] = ws[1] * f_scale;
            w_steps[i * 3 + 2] = ws[2] * f_scale;
            if (ex.w_ei > 0.0) wei_steps[i] = w_ei_at(t) * f_scale;
            if (rc.ema_decay > 0.0) {
                ++ema_updates;
                const double nom_dec = std::min(cur_ema_decay, (1.0 + (double)ema_updates) / (10.0 + (double)ema_updates));
                if (lazy_ema_copy_step > 0 && ema_updates < lazy_ema_copy_step) ema_dec[i] = 1.0;
                else if (lazy_ema_copy_step > 0 && ema_updates == lazy_ema_copy_step) ema_dec[i] = 0.0;
                else ema_dec[i] = nom_dec;
            }
        }
        LstBatches bt{spe, frame_ids.data()};
        auto te0 = Clock::now();
        CK_LST(h, lst_begin_epoch_steps(h, &bt, lr_steps.data(), w_steps.data(),
                                        rc.ema_decay > 0.0 ? ema_dec.data() : nullptr, st));
        if (ex.w_ei > 0.0) {
            CK_LST(h, lst_set_node_energy_steps(h, wei_steps.data(), spe, st));
        }
        CK_LST(h, lst_run_steps(h, spe, st));
        CK_LST(h, lst_epoch_losses(h, losses.data(), st));
        train_gpu_s += elapsed_s(te0);
        total_steps += spe;

        // Update importance-sampling frame scores
        if (rc.use_sampling) {
            double lmean = 0.0;
            for (int i = 0; i < spe; ++i) lmean += losses[i] / fac[i];
            lmean = std::max(lmean / (double)spe, 1e-300);
            const double sm = rc.smp_smooth;
            for (int i = 0; i < spe; ++i) {
                const double rel = (losses[i] / fac[i]) / lmean;
                for (int j = 0; j < bs; ++j) {
                    const int fid = frame_ids[i * bs + j];
                    score[fid] = (score[fid] < 0.0) ? rel : (sm * score[fid] + (1.0 - sm) * rel);
                }
            }
            have_scores = true;
            for (int f = 0; f < N_tr; ++f)
                if (score[f] < 0.0) { have_scores = false; break; }
        }

        if (rc.eval_every > 0 && ((ep % rc.eval_every == 0) || ep == rc.epochs - 1) && F_va > 0) {
            auto tv0 = Clock::now();
            auto vm = eval_range_ema(F_tr, F_tr + F_va);
            eval_s += elapsed_s(tv0);
            std::printf("  ep %2d/%d | train_s=%.3f | valid: E=%.3f meV/at  F=%.2f meV/A  S=%.3f meV/A^3  V=%.2f meV/at\n",
                        ep + 1, rc.epochs, train_gpu_s, vm.rmse_e_mev, vm.rmse_f_mev, vm.rmse_s_mev, vm.rmse_v_mev);
        }
    }
    const double t_loop = elapsed_s(t_tr0);

    // 5. Copy EMA -> params, optional Mode-1 VarPro polish, then final valid & test GPU evaluation
    if (rc.ema_decay > 0.0) {
        CK_CUDA(cudaMemcpyAsync(d_params_all, d_ema_all, tot_p * sizeof(T), cudaMemcpyDeviceToDevice, st));
    }
    if (ex.use_varpro) {
        varpro_s = run_varpro_t<T>(h, sh_tr, g_all, F_tr, m, d_params_all, d_ema_all, rc.ema_decay > 0.0,
                                   d_ev_e, d_ev_f, d_ev_s, ex.varpro_ridge, st);
    }
    auto t_ev0 = Clock::now();
    EvalMetrics vm{}, tm{};
    if (F_va > 0 || F_te > 0) {
        CK_LST(h, lst_set_eval_cache(h, 1));
        if (F_va > 0) vm = run_eval_t<T>(h, sh_tr, g_all, F_tr, F_tr + F_va, d_ev_e, d_ev_f, d_ev_s, st);
        if (F_te > 0) tm = run_eval_t<T>(h, sh_tr, g_all, F_tr + F_va, F_tr + F_va + F_te, d_ev_e, d_ev_f, d_ev_s, st);
        CK_LST(h, lst_set_eval_cache(h, 0));
    }
    const double t_final_eval = elapsed_s(t_ev0);

    // 6. Save trained EMA model (.lsm) with physical E0 restored
    auto t_sv0 = Clock::now();
    if (!out_path.empty()) {
        const T* d_src = (rc.ema_decay > 0.0) ? d_ema_all : d_params_all;
        if (m.dtype == 0) {
            CK_CUDA(cudaMemcpy(m.params_f32.data(), d_src, tot_p * sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            CK_CUDA(cudaMemcpy(m.params_f64.data(), d_src, tot_p * sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (did_center) m.E0 = physical_e0;
        save_lsm(out_path, m);
    }
    const double t_save = elapsed_s(t_sv0);
    const double t_total = elapsed_s(t_proc0);

    std::printf(
        "\n=== LOUPSTRAIN_CLI SUMMARY (%s, ep=%d, bs=%d%s, seed=%d, r_max=%.2f) ===\n"
        "  Timing : parse=%.3fs(bg=%.3fs) | gpu_prep=%.3fs | engine_init=%.3fs | train_gpu=%.3fs (%.3f ms/step) | "
        "varpro=%.3fs | eval=%.3fs | save=%.3fs | TOTAL_WALL=%.3fs\n",
        rc.name.c_str(), rc.epochs, rc.batch_size,
        (ex.ramp_ep > 0 && ex.ramp_b2 > 0) ? ("->b" + std::to_string(ex.ramp_b2)).c_str() : "",
        rc.seed, m.r_max, t_parse, t_parse_bg, t_prep, t_engine + ramp_s, train_gpu_s,
        1000.0 * train_gpu_s / (double)std::max<int64_t>(1, total_steps), varpro_s, eval_s + t_final_eval, t_save, t_total);
    if (F_va > 0) {
        std::printf("  VALID  : vE=%.3f meV/at | vF=%.2f meV/A | vS=%.3f meV/A^3 | vV=%.2f meV/at\n",
                    vm.rmse_e_mev, vm.rmse_f_mev, vm.rmse_s_mev, vm.rmse_v_mev);
    }
    if (F_te > 0) {
        std::printf("  TEST   : tE=%.3f meV/at | tF=%.2f meV/A | tS=%.3f meV/A^3 | tV=%.2f meV/at\n",
                    tm.rmse_e_mev, tm.rmse_f_mev, tm.rmse_s_mev, tm.rmse_v_mev);
    }
    if (F_va > 0 && F_te > 0) {
        const bool pass = (vm.rmse_f_mev <= 4.50 && vm.rmse_e_mev <= 0.040 &&
                           tm.rmse_f_mev <= 2.90 && tm.rmse_s_mev <= 0.50 && tm.rmse_v_mev <= 0.50);
        std::printf("  GATE   : %s (targets: vF<=4.50, vE<=0.040, tF<=2.90, tS/tV<=0.50)\n", pass ? "PASS" : "MISS");
    }

    // Cleanup
    if (d_ev_e) { cudaFree(d_ev_e); cudaFree(d_ev_f); cudaFree(d_ev_s); }
    lst_destroy(h);
    cudaFree(d_params_all); cudaFree(d_ema_all); cudaFree(d_tmp_p);
    cudaFree(d_m1_all); cudaFree(d_m2_all);
    if (d_vmax_all) cudaFree(d_vmax_all);
    cudaFree(d_steps_all);
    cudaStreamDestroy(st);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const auto t_proc0 = Clock::now();
    std::string init_path, train_path, valid_path, test_path, out_path, recipe_path;
    RecipeCfg rc;
    CliExtras ex;
    int ov_epochs = -1, ov_bs = -1, ov_seed = -1, ov_eval = -1, ov_kspline = -1, ov_soap_every = -1, reinit_seed = -1;
    double ov_cutoff = -1.0, ov_tail = -1.0, ov_ema = -1.0, ov_lr_peak = -1.0, ov_lr_final = -1.0, ov_lr_plateau = -1.0;
    double ov_ws = -1.0, ov_we = -1.0;
    bool center_e0 = false, auto_stats = false, dump_prep_check = false, no_tail = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--init") init_path = need("--init");
        else if (a == "--train") train_path = need("--train");
        else if (a == "--valid") valid_path = need("--valid");
        else if (a == "--test") test_path = need("--test");
        else if (a == "--out") out_path = need("--out");
        else if (a == "--recipe") recipe_path = need("--recipe");
        else if (a == "--epochs") ov_epochs = std::atoi(need("--epochs").c_str());
        else if (a == "--batch") ov_bs = std::atoi(need("--batch").c_str());
        else if (a == "--seed") ov_seed = std::atoi(need("--seed").c_str());
        else if (a == "--cutoff") ov_cutoff = std::atof(need("--cutoff").c_str());
        else if (a == "--tail-epoch") ov_tail = std::atof(need("--tail-epoch").c_str());
        else if (a == "--no-tail") no_tail = true;
        else if (a == "--ema-decay") ov_ema = std::atof(need("--ema-decay").c_str());
        else if (a == "--lr-peak") ov_lr_peak = std::atof(need("--lr-peak").c_str());
        else if (a == "--lr-final") ov_lr_final = std::atof(need("--lr-final").c_str());
        else if (a == "--lr-plateau") ov_lr_plateau = std::atof(need("--lr-plateau").c_str());
        else if (a == "--w-stress") ov_ws = std::atof(need("--w-stress").c_str());
        else if (a == "--w-energy") ov_we = std::atof(need("--w-energy").c_str());
        else if (a == "--w-ei") ex.w_ei = std::atof(need("--w-ei").c_str());
        else if (a == "--ei-decay-start") ex.ei_decay_start = std::atof(need("--ei-decay-start").c_str());
        else if (a == "--ei-decay-end") ex.ei_decay_end = std::atof(need("--ei-decay-end").c_str());
        else if (a == "--varpro") ex.use_varpro = true;
        else if (a == "--varpro-ridge") { ex.use_varpro = true; ex.varpro_ridge = std::atof(need("--varpro-ridge").c_str()); }
        else if (a == "--batch-ramp") {
            std::string spec = need("--batch-ramp");
            size_t col = spec.find(':');
            if (col == std::string::npos) { std::fprintf(stderr, "Bad --batch-ramp (want ep:b2): %s\n", spec.c_str()); return 1; }
            ex.ramp_ep = std::atoi(spec.substr(0, col).c_str());
            ex.ramp_b2 = std::atoi(spec.substr(col + 1).c_str());
        }
        else if (a == "--ramp-lr-mult") ex.ramp_lr_mult = std::atof(need("--ramp-lr-mult").c_str());
        else if (a == "--ramp-ema-decay") ex.ramp_ema_decay = std::atof(need("--ramp-ema-decay").c_str());
        else if (a == "--eval-every") ov_eval = std::atoi(need("--eval-every").c_str());
        else if (a == "--k-spline") ov_kspline = std::atoi(need("--k-spline").c_str());
        else if (a == "--soap-every") ov_soap_every = std::atoi(need("--soap-every").c_str());
        else if (a == "--reinit-seed") reinit_seed = std::atoi(need("--reinit-seed").c_str());
        else if (a == "--center-e0") center_e0 = true;
        else if (a == "--auto-stats") auto_stats = true;
        else if (a == "--prep-check") dump_prep_check = true;
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", a.c_str());
            return 1;
        }
    }
    if (init_path.empty() || train_path.empty()) {
        std::fprintf(stderr,
                     "Usage: %s --init init.lsm --train train.xyz [--valid valid.xyz] [--test test.xyz]\n"
                     "          [--recipe recipe.json] [--out trained.lsm] [--epochs N] [--batch B]\n"
                     "          [--seed S] [--cutoff R] [--center-e0] [--auto-stats] [--k-spline K]\n"
                     "          [--varpro] [--varpro-ridge R] [--no-tail] [--batch-ramp EP:B2]\n",
                     argv[0]);
        return 1;
    }

    if (!recipe_path.empty()) load_recipe_json(recipe_path, rc);
    if (ov_epochs > 0) {
        rc.epochs = ov_epochs;
        const double lrp = (ov_lr_peak > 0.0) ? ov_lr_peak : 0.05;
        const double lrf = (ov_lr_final > 0.0) ? ov_lr_final : (no_tail ? 0.0005 : 0.0001);
        const double ws0 = (ov_ws >= 0.0) ? ov_ws : 30.0;
        const double we0 = (ov_we >= 0.0) ? ov_we : 1.0;
        if (recipe_path.empty()) {
            rc.name = no_tail ? "notail" : "custom";
            if (no_tail) {
                if (ov_lr_plateau > 1.0 && ov_lr_plateau < (double)ov_epochs)
                    rc.lr_points = {{0.0, lrp * 0.1}, {1.0, lrp}, {ov_lr_plateau, lrp}, {(double)ov_epochs, lrf}};
                else
                    rc.lr_points = {{0.0, lrp * 0.1}, {1.0, lrp}, {(double)ov_epochs, lrf}};
                rc.weights = {{0.0, we0, 100.0, ws0, 0.0}};
            } else {
                const double tail_ep = (ov_tail >= 0.0) ? ov_tail : std::max((double)(ov_epochs - 4), std::floor(ov_epochs * 0.85));
                rc.lr_points = {{0.0, lrp * 0.1}, {1.0, lrp}, {tail_ep, 0.0015}, {(double)ov_epochs, lrf}};
                rc.weights = {{0.0, we0, 100.0, ws0, 0.0}, {tail_ep, 1000.0, 100.0, 10.0, 1.0}};
            }
        }
    }
    if (ov_ema >= 0.0) rc.ema_decay = ov_ema;
    if (ov_bs > 0) rc.batch_size = ov_bs;
    if (ov_seed >= 0) rc.seed = ov_seed;
    if (ov_eval >= 0) rc.eval_every = ov_eval;
    if (ov_soap_every > 0) rc.soap_every = ov_soap_every;

    // Set optimal Loup-S library defaults unless overridden by user environment
    if (ov_kspline > 0) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%d", ov_kspline);
        setenv("LST_MLP_SPLINE", buf, 1);
    } else {
        setenv("LST_MLP_SPLINE", "64", 0);
    }
    setenv("LST_SPLINE_HALF", "1", 0);
    setenv("LST_ACT_HALF", "2", 0);
    char sw_buf[16];
    std::snprintf(sw_buf, sizeof(sw_buf), "%d", rc.soap_sweeps);
    setenv("LST_SOAP_SWEEPS", sw_buf, 0);
    setenv("LST_SOAP_BLOCK", rc.soap_block ? "1" : "0", 0);

    LsmModel m = load_lsm(init_path);
    if (reinit_seed >= 0) {
        // Exact mace-torch ScaleShiftMACE random init: N(0,1) for e3nn linears/MLPs, N(0,1)/K_nu for symmetric contractions.
        std::mt19937_64 irng((uint64_t)reinit_seed * 6364136223846793005ULL + 1442695040888963407ULL);
        std::normal_distribution<double> nd(0.0, 1.0);
        const int K1 = m.K - m.K3 - m.K2;
        size_t off = 0;
        for (int i = 0; i < LST_N_PARAMS; ++i) {
            double sc = 1.0;
            if (i == 15 || i == 19) sc = 1.0 / (double)m.K3;
            else if (i == 16 || i == 20) sc = 1.0 / (double)m.K2;
            else if (i == 17 || i == 21) sc = 1.0 / (double)std::max(1, K1);
            for (int64_t j = 0; j < m.param_numel[i]; ++j) {
                const double v = nd(irng) * sc;
                if (m.dtype == 0) m.params_f32[off + (size_t)j] = (float)v;
                else m.params_f64[off + (size_t)j] = v;
            }
            off += (size_t)m.param_numel[i];
        }
    }
    if (ov_cutoff > 0.0) {
        // Update cutoff and Bessel basis weights (prefactor = sqrt(2/r_max), bessel_w[k] = (k+1)*pi/r_max)
        m.r_max = ov_cutoff;
        m.prefactor = std::sqrt(2.0 / ov_cutoff);
        for (int k = 0; k < m.NB; ++k) m.bessel_w[k] = (double)(k + 1) * M_PI / ov_cutoff;
    }

    if (m.dtype == 0) {
        return run_training_t<float>(m, rc, ex, train_path, valid_path, test_path, out_path,
                                     center_e0, auto_stats, dump_prep_check, t_proc0);
    } else {
        return run_training_t<double>(m, rc, ex, train_path, valid_path, test_path, out_path,
                                      center_e0, auto_stats, dump_prep_check, t_proc0);
    }
}
