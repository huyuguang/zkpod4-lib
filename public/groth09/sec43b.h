#pragma once

#include "./details.h"
#include "./sec53b.h"

// x, y, z: secret matrix<Fr>, size =m*n
// open: a1=com(gx, x1)...am=com(gx, xm)
// open b1=com(gy, y1)...bm=com(gy, ym)
// open c1=com(gz, z1)...cm=com(gz, zm)
// prove: z=x o y (o is hadamard product)
// proof size: 2*log(m)+6 G1, 3n+5 Fr
// prove cost: 2*log(m)*mulexp(n)
// verify cost: 2*mulexp(n)
namespace groth09::sec43b {

// input_53's z_g_offset can be any value, here just use input.z_g_offset or -1
int64_t SelectSec53Zoffset(int64_t x_g_offset, int64_t y_g_offset,
                           int64_t z_g_offset) {
  if (x_g_offset != -1 && y_g_offset != -1 && z_g_offset != -1) return -1;
  return z_g_offset;
}

struct CommitmentPub {
  std::vector<G1> a;  // a.size = m
  std::vector<G1> b;  // b.size = m
  std::vector<G1> c;  // c.size = m
  void Align() {
    int64_t old_m = a.size();
    int64_t new_m = (int64_t)misc::Pow2UB(old_m);
    if (new_m > old_m) {
      static const G1 g0 = G1Zero();
      a.resize(new_m);
      std::fill(a.begin() + old_m, a.end(), g0);
      b.resize(new_m);
      std::fill(b.begin() + old_m, b.end(), g0);
      c.resize(new_m);
      std::fill(c.begin() + old_m, c.end(), g0);
    }
  }
};

struct CommitmentSec {
  std::vector<Fr> r;  // r.size = m
  std::vector<Fr> s;  // s.size = m
  std::vector<Fr> t;  // t.size = m
  void Align() {
    int64_t old_m = r.size();
    int64_t new_m = (int64_t)misc::Pow2UB(old_m);
    if (new_m > old_m) {
      static const Fr f0 = FrZero();
      r.resize(new_m);
      std::fill(r.begin() + old_m, r.end(), f0);
      s.resize(new_m);
      std::fill(s.begin() + old_m, s.end(), f0);
      t.resize(new_m);
      std::fill(t.begin() + old_m, t.end(), f0);
    }
  }
};

struct RomProof {
  G1 c;
  sec53b::RomProof proof_53;     // 2*log(m)+4 G1, 2n+3 Fr
  hyrax::a2::RomProof proof_a2;  // 2 G1, n+2 Fr
  int64_t n() const { return proof_53.n(); }
  int64_t m() const { return proof_53.m(); }
};

inline bool operator==(RomProof const& left, RomProof const& right) {
  return left.c == right.c && left.proof_53 == right.proof_53 &&
         left.proof_a2 == right.proof_a2;
}
inline bool operator!=(RomProof const& left, RomProof const& right) {
  return !(left == right);
}

// save to bin
template <typename Ar>
void serialize(Ar& ar, RomProof const& t) {
  ar& YAS_OBJECT_NVP("43.rp", ("c", t.c), ("53p", t.proof_53),
                     ("a2p", t.proof_a2));
}

// load from bin
template <typename Ar>
void serialize(Ar& ar, RomProof& t) {
  ar& YAS_OBJECT_NVP("43.rp", ("c", t.c), ("53p", t.proof_53),
                     ("a2p", t.proof_a2));
}

struct ProverInput {
  std::vector<std::vector<Fr>> x;  // m*n
  std::vector<std::vector<Fr>> y;
  std::vector<std::vector<Fr>> z;
  int64_t const x_g_offset;
  int64_t const y_g_offset;
  int64_t const z_g_offset;

  int64_t m() const { return x.size(); }
  int64_t n() const { return x[0].size(); }
  void Take(std::vector<std::vector<Fr>>& ox, std::vector<std::vector<Fr>>& oy,
            std::vector<std::vector<Fr>>& oz) {
    ox = std::move(x);
    oy = std::move(y);
    oz = std::move(z);
  }

  ProverInput(std::vector<std::vector<Fr>> ix, std::vector<std::vector<Fr>> iy,
              std::vector<std::vector<Fr>> iz, int64_t x_g_offset,
              int64_t y_g_offset, int64_t z_g_offset)
      : x(std::move(ix)),
        y(std::move(iy)),
        z(std::move(iz)),
        x_g_offset(x_g_offset),
        y_g_offset(y_g_offset),
        z_g_offset(z_g_offset) {
    // Tick tick(__FUNCTION__);
    assert(!x.empty());
    assert(x.size() == y.size());
    assert(x.size() == z.size());
    for (auto i = 0LL; i < m(); ++i) {
      assert(x[i].size() == (size_t)n());
      assert(y[i].size() == (size_t)n());
      assert(z[i].size() == (size_t)n());
    }
  }

  // pad some trivial value
  void Align() {
    // Tick tick(__FUNCTION__);
    int64_t old_m = m();
    int64_t new_m = (int64_t)misc::Pow2UB(old_m);
    if (old_m == new_m) return;

    auto const& f0 = FrZero();
    x.resize(new_m);
    y.resize(new_m);
    z.resize(new_m);
    for (int64_t i = old_m; i < new_m; ++i) {
      auto& xi = x[i];
      xi.resize(n());
      std::fill(xi.begin(), xi.end(), f0);
      auto& yi = y[i];
      yi.resize(n());
      std::fill(yi.begin(), yi.end(), f0);
      auto& zi = z[i];
      zi.resize(n());
      std::fill(zi.begin(), zi.end(), f0);
    }
  }
};

inline void ComputeCom(CommitmentPub& com_pub, CommitmentSec& com_sec,
                       ProverInput const& input) {
  // Tick tick(__FUNCTION__);
  auto const m = input.m();
  com_sec.r.resize(m);
  FrRand(com_sec.r.data(), m);

  com_sec.s.resize(m);
  FrRand(com_sec.s.data(), m);

  com_sec.t.resize(m);
  FrRand(com_sec.t.data(), m);

  com_pub.a.resize(m);
  com_pub.b.resize(m);
  com_pub.c.resize(m);

  auto parallel_f = [&com_sec, &com_pub, &input](int64_t i) {
    com_pub.a[i] =
        PcComputeCommitmentG(input.x_g_offset, input.x[i], com_sec.r[i]);
    com_pub.b[i] =
        PcComputeCommitmentG(input.y_g_offset, input.y[i], com_sec.s[i]);
    com_pub.c[i] =
        PcComputeCommitmentG(input.z_g_offset, input.z[i], com_sec.t[i]);
  };
  parallel::For(m, parallel_f);
}

inline void UpdateSeed(h256_t& seed, CommitmentPub const& com_pub) {
  // Tick tick(__FUNCTION__);
  CryptoPP::Keccak_256 hash;
  HashUpdate(hash, seed);
  HashUpdate(hash, com_pub.a);
  HashUpdate(hash, com_pub.b);
  HashUpdate(hash, com_pub.c);
  hash.Final(seed.data());
}

inline void ComputeChallengeKT(h256_t const& seed, std::vector<Fr>& k,
                               std::vector<Fr>& t) {
  ComputeFst(seed, "gro09::sec43b::k", k);
  ComputeFst(seed, "gro09::sec43b::t", t);
}

// pad some trivial value
inline void AlignData(ProverInput& input, CommitmentPub& com_pub,
                      CommitmentSec& com_sec) {
  // Tick tick(__FUNCTION__);
  input.Align();
  com_pub.Align();
  com_sec.Align();
}

inline void RomProve(RomProof& rom_proof, h256_t seed, ProverInput input,
                     CommitmentPub com_pub, CommitmentSec com_sec) {
  // Tick tick(__FUNCTION__);
  auto m = input.m();
  auto n = input.n();

  UpdateSeed(seed, com_pub);
  std::vector<Fr> k(m);
  std::vector<Fr> t(n);
  ComputeChallengeKT(seed, k, t);

  Fr input_53_z;
  G1 com_pub_53_c;
  Fr com_sec_53_t;

  std::vector<std::vector<Fr>> input_x;
  std::vector<std::vector<Fr>> input_y;
  std::vector<std::vector<Fr>> input_z;
  input.Take(input_x, input_y, input_z);

  {
    // Tick tick53(" prepare for sec53b");

    sec53b::CommitmentSec com_sec_53;
    sec53b::CommitmentPub com_pub_53;
    std::vector<std::vector<Fr>> x_53(m);

    auto parallel_f = [&input_x, &k](int64_t i) { input_x[i] *= k[i]; };
    parallel::For(m, parallel_f, m < 1024);

    std::vector<std::vector<Fr>> input_yt(m);
    Fr z = FrZero();
    // TODO: parallel
    for (int64_t i = 0; i < m; ++i) {
      input_yt[i] = details::HadamardProduct(input_y[i], t);
      z += InnerProduct(input_x[i], input_yt[i]);
    }

    auto input_53_z_g_offset = SelectSec53Zoffset(
        input.x_g_offset, input.y_g_offset, input.z_g_offset);
    sec53b::ProverInput input_53(std::move(input_x), std::move(input_y), t,
                                 std::move(input_yt), z, input.x_g_offset,
                                 input.y_g_offset, input_53_z_g_offset);
    input_53_z = input_53.z;

    com_sec_53.r.resize(m);
    com_pub_53.a.resize(m);
    auto parallel_f2 = [&com_sec, &com_pub, &com_sec_53, &com_pub_53,
                        &k](int64_t i) {
      com_sec_53.r[i] = com_sec.r[i] * k[i];
      com_pub_53.a[i] = com_pub.a[i] * k[i];
    };
    parallel::For(m, parallel_f2, m < 16 * 1024);

    com_sec_53.s = com_sec.s;
    com_sec_53.t = FrRand();
    com_sec_53_t = com_sec_53.t;

    com_pub_53.b = com_pub.b;
    com_pub_53.c =
        PcComputeCommitmentG(input_53.z_g_offset, input_53_z, com_sec_53.t);
    rom_proof.c = com_pub_53.c;  // verifier can not compute c by com_pub.c
    com_pub_53_c = com_pub_53.c;

    sec53b::RomProve(rom_proof.proof_53, seed, std::move(input_53),
                     std::move(com_pub_53), std::move(com_sec_53));
  }

  {
    // Tick tick53("prepare for hyrax");
    hyrax::a2::CommitmentPub com_pub_hy;
    hyrax::a2::CommitmentSec com_sec_hy;

    std::vector<Fr> x_hy(n);

    std::fill(x_hy.begin(), x_hy.end(), FrZero());

    auto parallel_f = [&input_z, &x_hy, &k, m](int64_t j) mutable {
      for (int64_t i = 0; i < m; ++i) {
        x_hy[j] += input_z[i][j] * k[i];
      }
    };
    parallel::For(n, parallel_f, n < 16 * 1024);

    auto input_a2_x_g_offset = input.z_g_offset;
    auto input_a2_y_g_offset = SelectSec53Zoffset(
        input.x_g_offset, input.y_g_offset, input.z_g_offset);

    hyrax::a2::ProverInput input_hy(x_hy, t, input_53_z, input_a2_x_g_offset,
                                    input_a2_y_g_offset);

    com_sec_hy.r_xi = InnerProduct(com_sec.t, k);
    com_sec_hy.r_tau = com_sec_53_t;

    com_pub_hy.tau = com_pub_53_c;

    // do not need to compute the com1_pub_hy.xi in release build
    assert(input_hy.y == InnerProduct(input_hy.x, input_hy.a));

    com_pub_hy.xi = MultiExpBdlo12(com_pub.c, k);
#ifdef _DEBUG
    auto check_xi =
        PcComputeCommitmentG(input.z_g_offset, input_hy.x, com_sec_hy.r_xi);
    assert(check_xi == com_pub_hy.xi);
#endif

    hyrax::a2::RomProve(rom_proof.proof_a2, seed, std::move(input_hy),
                        std::move(com_pub_hy), std::move(com_sec_hy));
  }
}

struct VerifierInput {
  VerifierInput(CommitmentPub const& com_pub, int64_t x_g_offset,
                int64_t y_g_offset, int64_t z_g_offset)
      : com_pub(com_pub),
        x_g_offset(x_g_offset),
        y_g_offset(y_g_offset),
        z_g_offset(z_g_offset) {}
  CommitmentPub const& com_pub;
  int64_t const x_g_offset;
  int64_t const y_g_offset;
  int64_t const z_g_offset;
};

inline bool RomVerify(RomProof const& rom_proof, h256_t seed,
                      VerifierInput const& input) {
  // Tick tick(__FUNCTION__);
  auto m = rom_proof.m();
  auto n = rom_proof.n();

  auto const& com_pub = input.com_pub;
  UpdateSeed(seed, com_pub);
  std::vector<Fr> k(m);
  std::vector<Fr> t(n);
  ComputeChallengeKT(seed, k, t);

  std::array<parallel::Task, 2> tasks;
  bool ret_53 = false;
  tasks[0] = [&ret_53, &rom_proof, &input, m, &com_pub, &k, &t, &seed]() {
    sec53b::CommitmentPub com_pub_53;
    com_pub_53.c = rom_proof.c;
    com_pub_53.b = input.com_pub.b;
    com_pub_53.a.resize(m);
    auto parallel_f = [&com_pub_53, &com_pub, &k](int64_t i) {
      com_pub_53.a[i] = com_pub.a[i] * k[i];
    };
    parallel::For(m, parallel_f, m < 1024);

    auto intput_53_z_g_offset = SelectSec53Zoffset(
        input.x_g_offset, input.y_g_offset, input.z_g_offset);

    sec53b::VerifierInput input_53(t, com_pub_53, input.x_g_offset,
                                   input.y_g_offset, intput_53_z_g_offset);
    ret_53 = sec53b::RomVerify(rom_proof.proof_53, seed, input_53);
    assert(ret_53);
  };

  bool ret_a2 = false;
  tasks[1] = [&ret_a2, &com_pub, &rom_proof, &t, &k, &seed, &input]() {
    hyrax::a2::CommitmentPub com_pub_hy(MultiExpBdlo12(com_pub.c, k),
                                        rom_proof.c);
    auto input_a2_x_g_offset = input.z_g_offset;
    auto input_a2_y_g_offset = SelectSec53Zoffset(
        input.x_g_offset, input.y_g_offset, input.z_g_offset);
    hyrax::a2::VerifierInput input_hy(t, com_pub_hy, input_a2_x_g_offset,
                                      input_a2_y_g_offset);
    ret_a2 = hyrax::a2::RomVerify(rom_proof.proof_a2, seed, input_hy);
    assert(ret_a2);
  };

  parallel::Invoke(tasks);

  if (!ret_53 || !ret_a2) {
    std::cout << "ret_53: " << ret_53 << ", ret_a2: " << ret_a2 << "\n";
    assert(false);
  }
  return ret_53 && ret_a2;
}

inline bool TestRom(int64_t m, int64_t n) {
  std::cout << "old_m=" << m << ", n=" << n << "\n";

  std::vector<std::vector<Fr>> x(m);
  for (auto& i : x) {
    i.resize(n);
    FrRand(i.data(), n);
  }

  std::vector<std::vector<Fr>> y(m);
  for (auto& i : y) {
    i.resize(n);
    FrRand(i.data(), n);
  }

  std::vector<std::vector<Fr>> z(m);
  for (int64_t i = 0; i < m; ++i) {
    z[i] = details::HadamardProduct(x[i], y[i]);
  }

  h256_t seed = misc::RandH256();

  int64_t x_g_offset = 0;
  int64_t y_g_offset = 0;
  int64_t z_g_offset = 0;

  ProverInput prover_input(x, y, z, x_g_offset, y_g_offset, z_g_offset);
  CommitmentPub com_pub;
  CommitmentSec com_sec;
  ComputeCom(com_pub, com_sec, prover_input);

  AlignData(prover_input, com_pub, com_sec);

  RomProof rom_proof;
  RomProve(rom_proof, seed, prover_input, com_pub, com_sec);

  VerifierInput verifier_input(com_pub, x_g_offset, y_g_offset, z_g_offset);
  bool success = RomVerify(rom_proof, seed, verifier_input);
  std::cout << __FILE__ << " " << __FUNCTION__ << ": " << success << "\n";
  return success;
}
}  // namespace groth09::sec43b