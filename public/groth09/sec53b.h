#pragma once

#include <map>
#include <memory>

#include "groth09/details.h"
#include "groth09/sec51b.h"
#include "groth09/sec51c.h"
#include "utils/fst.h"

// t: public vector<Fr>, size = n
// x, y: secret matric<Fr>, size = m*n
// z: secret Fr
// open: com(gx,x1), com(gy,y1), com(gx,x2), com(gy,y2) ... com(gz,z)
// prove: z = <x1,y1 o t> + <x2,y2 o t>...
// proof size: 2*log(m)+4 G1, 2n+3 Fr
// prove cost: 2*log(m)*mulexp(n)
// verify cost: mulexp(n)
// base on Sec51

namespace groth09 {

// Sec51: Sec51B or Sec51C
template <typename Sec51>
struct Sec53b {
  struct CommitmentPub {
    std::vector<G1> a;  // a.size = m
    std::vector<G1> b;  // b.size = m
    G1 c;
    int64_t m() const { return a.size(); }
    bool CheckFormat() const {
      if (a.empty() || a.size() != b.size()) return false;
      return misc::Pow2UB(m()) == (uint64_t)m();
    }
    void Align() {
      int64_t old_m = a.size();
      int64_t new_m = (int64_t)misc::Pow2UB(old_m);
      if (new_m > old_m) {
        a.resize(new_m, G1Zero());
        b.resize(new_m, G1Zero());
      }
    }
  };

  struct CommitmentSec {
    std::vector<Fr> r;  // r.size = m
    std::vector<Fr> s;  // s.size = m
    Fr t;
    void Align() {
      int64_t old_m = r.size();
      int64_t new_m = (int64_t)misc::Pow2UB(old_m);
      if (new_m > old_m) {
        r.resize(new_m, FrZero());
        s.resize(new_m, FrZero());
      }
    }
  };

  struct VerifyInput {
    VerifyInput(std::vector<Fr> const& t, CommitmentPub const& com_pub,
                GetRefG1 const& get_gx, GetRefG1 const& get_gy, G1 const& gz)
        : t(t), com_pub(com_pub), get_gx(get_gx), get_gy(get_gy), gz(gz) {}
    std::vector<Fr> const& t;  // size = n
    CommitmentPub const& com_pub;
    int64_t m() const { return com_pub.m(); }
    bool CheckFormat(int64_t check_n) const {
      if (t.size() != (uint64_t)check_n) return false;
      return com_pub.CheckFormat();
    }
    GetRefG1 const& get_gx;
    GetRefG1 const& get_gy;
    G1 const& gz;
  };

  struct ProveInput {
    std::vector<std::vector<Fr>> x;
    std::vector<std::vector<Fr>> y;
    std::vector<Fr> const& t;
    std::vector<std::vector<Fr>> yt;
    Fr z;
    GetRefG1 const& get_gx;
    GetRefG1 const& get_gy;
    G1 const& gz;
    size_t n_ = 0;

    int64_t m() const { return x.size(); }
    int64_t n() const { return n_; }

    ProveInput(std::vector<std::vector<Fr>>&& ix,
               std::vector<std::vector<Fr>>&& iy, std::vector<Fr> const& it,
               std::vector<std::vector<Fr>>&& iyt, Fr const& z,
               GetRefG1 const& get_gx, GetRefG1 const& get_gy, G1 const& gz)
        : x(std::move(ix)),
          y(std::move(iy)),
          t(it),
          yt(std::move(iyt)),
          z(z),
          get_gx(get_gx),
          get_gy(get_gy),
          gz(gz) {
      if (x.empty()) {
        std::cout << __FN__ << ":" << __LINE__ << "oops\n";
        throw std::runtime_error("oops");
      }

      if (x.size() != y.size() || x.size() != yt.size()) {
        std::cout << __FN__ << ":" << __LINE__ << "oops\n";
        throw std::runtime_error("oops");
      }

      for (int64_t i = 0; i < m(); ++i) {
        if (x[i].size() != y[i].size() || x[i].size() != yt[i].size()) {
          std::cout << __FN__ << ":" << __LINE__ << "oops\n";
          throw std::runtime_error("oops");
        }
        if (x[i].size() > n_) n_ = x[i].size();
      }

      if (t.size() != n_) {
        std::cout << __FN__ << ":" << __LINE__ << "oops\n";
        throw std::runtime_error("oops");
      }

#ifdef _DEBUG
      Fr check_z = FrZero();
      for (int64_t i = 0; i < m(); ++i) {
        assert(yt[i] == HadamardProduct(y[i], t));
        check_z += InnerProduct(x[i], yt[i]);
      }
      assert(z == check_z);
#endif
    }

    void Align() {
      // Tick tick(__FN__);
      int64_t old_m = m();
      int64_t new_m = (int64_t)misc::Pow2UB(old_m);
      if (old_m < new_m) {
        x.resize(new_m);
        y.resize(new_m);
        yt.resize(new_m);
      }
    }

    void Update(Fr const& sigma_xy1, Fr const& sigma_xy2, Fr const& e,
                Fr const& ee) {
      Tick tick(__FN__, std::to_string(m()) + "," + std::to_string(n()));
      auto m2 = m() / 2;

      {
        std::vector<std::vector<Fr>> x2(m2);
        std::vector<std::vector<Fr>> y2(m2);

        auto pf1 = [this, &x2, &y2, &e](int64_t i) {
          x2[i] = x[2 * i + 1] * e + x[2 * i];
          y2[i] = y[2 * i] * e + y[2 * i + 1];
        };
        parallel::For(m2, pf1, n() < 1024);
        x2.swap(x);
        y2.swap(y);
      }

      {
        std::vector<std::vector<Fr>> yt2(m2);
        auto parallel_f = [this, &yt2](int64_t i) {
          yt2[i] = HadamardProduct(y[i], t);
        };
        parallel::For(m2, parallel_f, n() < 1024);
        yt2.swap(yt);
      }

      z = sigma_xy1 * ee + z * e + sigma_xy2;
    }
  };

  struct CommitmentExtPub {
    // 5.3 Recursive
    std::vector<G1> cl;  // size = log(m)
    std::vector<G1> cu;  // size = log(m)
    bool CheckFormat(int64_t check_m) const {
      if (cl.size() != cu.size()) return false;
      return m() == check_m;
    }
    int64_t m() const { return 1LL << cl.size(); }
    bool operator==(CommitmentExtPub const& right) const {
      return cl == right.cl && cu == right.cu;
    }

    bool operator!=(CommitmentExtPub const& right) const {
      return !(*this == right);
    }
    template <typename Ar>
    void serialize(Ar& ar) const {
      ar& YAS_OBJECT_NVP("53b.cep", ("cl", cl), ("cu", cu));
    }
    template <typename Ar>
    void serialize(Ar& ar) {
      ar& YAS_OBJECT_NVP("53b.cep", ("cl", cl), ("cu", cu));
    }
  };

  struct Proof {
    CommitmentExtPub com_ext_pub;    // 2*log(m) G1
    typename Sec51::Proof proof_51;  // 4 G1, 2n+3 Fr

    bool CheckFormat(int64_t check_m) const {
      return com_ext_pub.CheckFormat(check_m) && proof_51.CheckFormat();
    }
    bool operator==(Proof const& right) const {
      return com_ext_pub == right.com_ext_pub && proof_51 == right.proof_51;
    }

    bool operator!=(Proof const& right) const { return !(*this == right); }

    template <typename Ar>
    void serialize(Ar& ar) const {
      ar& YAS_OBJECT_NVP("53.pf", ("c", com_ext_pub), ("r", proof_51));
    }
    template <typename Ar>
    void serialize(Ar& ar) {
      ar& YAS_OBJECT_NVP("53.pf", ("c", com_ext_pub), ("r", proof_51));
    }
  };

  static void ComputeCom(ProveInput const& input, CommitmentPub* com_pub,
                         CommitmentSec const& com_sec) {
    Tick tick(__FN__);
    auto const m = input.m();
    auto const n = input.n();

    com_pub->a.resize(m);
    com_pub->b.resize(m);

    auto parallel_f = [&input, &com_pub, &com_sec](int64_t i) {
      com_pub->a[i] = pc::ComputeCom(input.get_gx, input.x[i], com_sec.r[i]);
      com_pub->b[i] = pc::ComputeCom(input.get_gy, input.y[i], com_sec.s[i]);
    };
    parallel::For(m, parallel_f, m * n < 16 * 1024);

    com_pub->c = pc::ComputeCom(input.gz, input.z, com_sec.t);
  }

  static void ComputeCom(ProveInput const& input, CommitmentPub* com_pub,
                         CommitmentSec* com_sec) {
    Tick tick(__FN__);
    auto const m = input.m();
    com_sec->r.resize(m);
    FrRand(com_sec->r.data(), m);

    com_sec->s.resize(m);
    FrRand(com_sec->s.data(), m);

    com_sec->t = FrRand();

    ComputeCom(input, com_pub, *com_sec);
  }

  static void ProveFinal(Proof& proof, h256_t const& seed,
                         ProveInput const& input, CommitmentPub const& com_pub,
                         CommitmentSec const& com_sec) {
    Tick tick(__FN__);
    assert(input.m() == 1);

    typename Sec51::ProveInput input_51(input.x[0], input.y[0], input.t,
                                        input.yt[0], input.z, input.get_gx,
                                        input.get_gy, input.gz);

    typename Sec51::CommitmentPub com_pub_51(com_pub.a[0], com_pub.b[0],
                                             com_pub.c);
    typename Sec51::CommitmentSec com_sec_51(com_sec.r[0], com_sec.s[0],
                                             com_sec.t);
    Sec51::Prove(proof.proof_51, seed, input_51, com_pub_51, com_sec_51);
  }

  static void ComputeSigmaXY(ProveInput const& input, Fr* sigma_xy1,
                             Fr* sigma_xy2) {
    Tick tick(__FN__);
    int64_t m = input.m();
    int64_t n = input.n();
    auto m2 = m / 2;
    std::vector<Fr> xy1(m2, FrZero());
    std::vector<Fr> xy2(m2, FrZero());
    auto parallel_f = [&input, &xy1, &xy2](int64_t i) {
      auto const& x1 = input.x[2 * i + 1];
      auto const& yt1 = input.yt[2 * i];
      xy1[i] = InnerProduct(x1, yt1);

      auto const& x2 = input.x[2 * i];
      auto const& yt2 = input.yt[2 * i + 1];
      xy2[i] = InnerProduct(x2, yt2);
    };
    parallel::For(m2, parallel_f, m * n < 16 * 1024);

    *sigma_xy1 = parallel::Accumulate(xy1.begin(), xy1.end(), FrZero());
    *sigma_xy2 = parallel::Accumulate(xy2.begin(), xy2.end(), FrZero());
  }

  static void UpdateCom(CommitmentPub& com_pub, CommitmentSec& com_sec,
                        Fr const& tl, Fr const& tu, G1 const& cl, G1 const& cu,
                        Fr const& e, Fr const& ee) {
    // Tick tick(__FN__);
    CommitmentPub com_pub2;
    CommitmentSec com_sec2;
    auto m2 = com_pub.a.size() / 2;
    com_pub2.a.resize(m2);
    com_pub2.b.resize(m2);
    com_sec2.r.resize(m2);
    com_sec2.s.resize(m2);

    auto parallel_f = [&com_pub, &com_sec, &com_pub2, &com_sec2,
                       &e](int64_t i) {
      auto& a2 = com_pub2.a;
      auto const& a = com_pub.a;
      a2[i] = a[2 * i] + a[2 * i + 1] * e;

      auto& b2 = com_pub2.b;
      auto const& b = com_pub.b;
      b2[i] = b[2 * i] * e + b[2 * i + 1];

      auto& r2 = com_sec2.r;
      auto const& r = com_sec.r;
      r2[i] = r[2 * i] + r[2 * i + 1] * e;

      auto& s2 = com_sec2.s;
      auto const& s = com_sec.s;
      s2[i] = s[2 * i] * e + s[2 * i + 1];
    };
    parallel::For((int64_t)m2, parallel_f);

    com_pub2.c = cl * ee + com_pub.c * e + cu;
    com_sec2.t = tl * ee + com_sec.t * e + tu;
    com_pub = std::move(com_pub2);
    com_sec = std::move(com_sec2);
  }

  static Fr ComputeChallenge(h256_t const& seed, CommitmentPub const& com_pub,
                             G1 const& cl, G1 const& cu) {
    // Tick tick(__FN__);
    CryptoPP::Keccak_256 hash;
    h256_t digest;
    HashUpdate(hash, seed);
    HashUpdate(hash, cl);
    HashUpdate(hash, cu);
    HashUpdate(hash, com_pub.a);
    HashUpdate(hash, com_pub.b);
    HashUpdate(hash, com_pub.c);
    hash.Final(digest.data());
    return H256ToFr(digest);
  }

  // pad some trivial value
  static void AlignData(ProveInput& input, CommitmentPub& com_pub,
                        CommitmentSec& com_sec) {
    // Tick tick(__FN__);
    input.Align();
    com_sec.Align();
    com_pub.Align();
  }

  static void ProveRecursive(Proof& proof, h256_t& seed, ProveInput& input,
                             CommitmentPub& com_pub, CommitmentSec& com_sec) {
    Tick tick(__FN__);
    assert(input.m() > 1);

    Fr sigma_xy1, sigma_xy2;
    ComputeSigmaXY(input, &sigma_xy1, &sigma_xy2);

    // compute cl, cu
    Fr tl = FrRand();
    Fr tu = FrRand();
    G1 cl = pc::ComputeCom(input.gz, sigma_xy1, tl);
    G1 cu = pc::ComputeCom(input.gz, sigma_xy2, tu);
    proof.com_ext_pub.cl.push_back(cl);
    proof.com_ext_pub.cu.push_back(cu);

    // challenge
    Fr e = ComputeChallenge(seed, com_pub, cl, cu);
    Fr ee = e * e;
    seed = FrToBin(e);

    input.Update(sigma_xy1, sigma_xy2, e, ee);

    UpdateCom(com_pub, com_sec, tl, tu, cl, cu, e, ee);
    // debug check com_pub2 and com_sec2
#ifdef _DEBUG
    CommitmentPub check_com_pub;
    ComputeCom(input, &check_com_pub, com_sec);
    assert(check_com_pub.a == com_pub.a);
    assert(check_com_pub.b == com_pub.b);
    assert(check_com_pub.c == com_pub.c);
#endif
  }

  static void Prove(Proof& proof, h256_t seed, ProveInput input,
                    CommitmentPub com_pub, CommitmentSec com_sec) {
    Tick tick(__FN__);
    assert(pc::Base::kGSize >= input.n());

    while (input.m() > 1) {
      ProveRecursive(proof, seed, input, com_pub, com_sec);
    }
    return ProveFinal(proof, seed, input, com_pub, com_sec);
  }

  static bool Verify(Proof const& proof, h256_t seed,
                     VerifyInput const& input) {
    Tick tick(__FN__);
    if (!proof.CheckFormat(input.m())) {
      assert(false);
      return false;
    }

    CommitmentPub com_pub = input.com_pub;

    for (size_t loop = 0; loop < proof.com_ext_pub.cl.size(); ++loop) {
      // challenge
      auto const& cl = proof.com_ext_pub.cl[loop];
      auto const& cu = proof.com_ext_pub.cu[loop];
      Fr e = ComputeChallenge(seed, com_pub, cl, cu);
      Fr ee = e * e;
      seed = FrToBin(e);

      std::vector<G1> a2(com_pub.m() / 2);
      std::vector<G1> b2(com_pub.m() / 2);
      G1 c2;

      auto m2 = com_pub.m() / 2;
      auto parallel_f = [&com_pub, &a2, &b2, &e](int64_t i) {
        auto const& a = com_pub.a;
        a2[i] = a[2 * i] + a[2 * i + 1] * e;

        auto const& b = com_pub.b;
        b2[i] = b[2 * i] * e + b[2 * i + 1];
      };
      parallel::For(m2, parallel_f, m2 < 1024);

      c2 = cl * ee + com_pub.c * e + cu;

      com_pub.a = std::move(a2);
      com_pub.b = std::move(b2);
      com_pub.c = std::move(c2);
    }

    assert(com_pub.m() == 1);

    typename Sec51::CommitmentPub com_pub_51(com_pub.a[0], com_pub.b[0],
                                             com_pub.c);
    typename Sec51::VerifyInput verifier_input_51(
        input.t, com_pub_51, input.get_gx, input.get_gy, input.gz);
    return Sec51::Verify(proof.proof_51, seed, verifier_input_51);
  }

  static bool Test(int64_t m, int64_t n);
};

template <typename Sec51>
bool Sec53b<Sec51>::Test(int64_t m, int64_t n) {
  Tick tick(__FN__);
  std::cout << "m=" << m << ", n=" << n << "\n";

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

  x.resize(x.size() + 1);
  x.back().resize(n / 2);
  FrRand(x.back());

  y.resize(y.size() + 1);
  y.back().resize(n / 2);
  FrRand(y.back());
  ++m;

  std::vector<Fr> t(n);
  FrRand(t.data(), t.size());

  std::vector<std::vector<Fr>> yt(m);
  Fr z = FrZero();
  for (int64_t i = 0; i < m; ++i) {
    yt[i] = HadamardProduct(y[i], t);
    z += InnerProduct(x[i], yt[i]);
  }

  h256_t seed = misc::RandH256();

  int64_t x_g_offset = 220;
  int64_t y_g_offset = 450;
  GetRefG1 get_gx = [x_g_offset](int64_t i) -> G1 const& {
    return pc::PcG()[x_g_offset + i];
  };
  GetRefG1 get_gy = [y_g_offset](int64_t i) -> G1 const& {
    return pc::PcG()[y_g_offset + i];
  };

  ProveInput prove_input(std::move(x), std::move(y), t, std::move(yt), z,
                         get_gx, get_gy, pc::PcU());
  CommitmentPub com_pub;
  CommitmentSec com_sec;
  ComputeCom(prove_input, &com_pub, &com_sec);

  AlignData(prove_input, com_pub, com_sec);

  Proof proof;
  Prove(proof, seed, prove_input, com_pub, com_sec);

#ifndef DISABLE_SERIALIZE_CHECK
  // serialize to buffer
  yas::mem_ostream os;
  yas::binary_oarchive<yas::mem_ostream, YasBinF()> oa(os);
  oa.serialize(proof);
  std::cout << "proof size: " << os.get_shared_buffer().size << "\n";
  // serialize from buffer
  yas::mem_istream is(os.get_intrusive_buffer());
  yas::binary_iarchive<yas::mem_istream, YasBinF()> ia(is);
  Proof proof2;
  ia.serialize(proof2);
  if (proof != proof2) {
    assert(false);
    std::cout << "oops, serialize check failed\n";
    return false;
  }
#endif

  VerifyInput verify_input(t, com_pub, get_gx, get_gy, pc::PcU());
  bool success = Verify(proof, seed, verify_input);
  std::cout << __FILE__ << " " << __FN__ << ": " << success << "\n\n\n\n\n\n";
  return success;
}
}  // namespace groth09