#pragma once

#include "./details.h"

// a: public vector<Fr>, size = n
// x: secret vector<Fr>, size = n
// y: secret Fr, = <x,a>
// open: com(gx,x), com(gy,y)
// prove: y=<x,a>
// proof size: 2 G1 and n+2 Fr
// prove cost: mulexp(n)
// verify cost: mulexp(n)
namespace hyrax {
struct A2 {
  struct ProverInput {
    ProverInput(std::vector<Fr> const& x, std::vector<Fr> const& a, Fr const& y,
                int64_t x_g_offset, int64_t y_g_offset)
        : x(x), a(a), y(y), x_g_offset(x_g_offset), y_g_offset(y_g_offset) {
      assert(y == InnerProduct(x, a));
    }
    int64_t n() const { return (int64_t)x.size(); }
    std::vector<Fr> const& x;  // x.size = n
    std::vector<Fr> const& a;  // a.size = n
    Fr const y;                // y = <x, a>
    int64_t const x_g_offset;
    int64_t const y_g_offset;
  };

  struct CommitmentPub {
    CommitmentPub() {}
    CommitmentPub(G1 const& xi, G1 const& tau) : xi(xi), tau(tau) {}
    G1 xi;   // com(gx, x, r_xi)
    G1 tau;  // com(gy, y, r_tau)
    bool operator==(CommitmentPub const& right) const {
      return xi == right.xi && tau == right.tau;
    }

    bool operator!=(CommitmentPub const& right) const {
      return !(*this == right);
    }
  };

  struct CommitmentSec {
    CommitmentSec() {}
    CommitmentSec(Fr const& x, Fr const& t) : r_xi(x), r_tau(t) {}
    Fr r_xi;
    Fr r_tau;
  };

  struct CommitmentExtPub {
    CommitmentExtPub() {}
    CommitmentExtPub(G1 const& delta, G1 const& beta)
        : delta(delta), beta(beta) {}
    G1 delta;  // com(d, r_delta)
    G1 beta;   // com(<a,d>, r_beta)
    bool operator==(CommitmentExtPub const& right) const {
      return delta == right.delta && beta == right.beta;
    }

    bool operator!=(CommitmentExtPub const& right) const {
      return !(*this == right);
    }
  };

  struct CommitmentExtSec {
    std::vector<Fr> d;  // size = n
    Fr r_beta;
    Fr r_delta;
  };

  struct SubProof {
    std::vector<Fr> z;  // z.size = n
    Fr z_delta;
    Fr z_beta;
    int64_t n() const { return (int64_t)z.size(); }
    bool operator==(SubProof const& right) const {
      return z == right.z && z_delta == right.z_delta && z_beta == right.z_beta;
    }

    bool operator!=(SubProof const& right) const { return !(*this == right); }
  };

  struct Proof {
    CommitmentExtPub com_ext_pub;  // 2 G1
    SubProof sub_proof;            // n+2 Fr
    int64_t n() const { return sub_proof.n(); }
    bool operator==(Proof const& right) const {
      return com_ext_pub == right.com_ext_pub && sub_proof == right.sub_proof;
    }

    bool operator!=(Proof const& right) const { return !(*this == right); }
  };

  struct VerifierInput {
    VerifierInput(std::vector<Fr> const& a, CommitmentPub const& com_pub,
                  int64_t x_g_offset, int64_t y_g_offset)
        : a(a),
          com_pub(com_pub),
          x_g_offset(x_g_offset),
          y_g_offset(y_g_offset) {}
    std::vector<Fr> const& a;  // a.size = n
    CommitmentPub const& com_pub;
    int64_t const x_g_offset;
    int64_t const y_g_offset;
  };

  // com(n) + com(1) + ip(n)
  static bool VerifyInternal(VerifierInput const& input, Fr const& challenge,
                             CommitmentExtPub const& com_ext_pub,
                             SubProof const& sub_proof) {
    // Tick tick(__FUNCTION__);
    auto const& com_pub = input.com_pub;

    std::array<parallel::Task, 2> tasks;
    bool ret1 = false;
    tasks[0] = [&ret1, &com_pub, &com_ext_pub, &challenge, &sub_proof,
                &input]() {
      auto const& xi = com_pub.xi;
      auto const& delta = com_ext_pub.delta;
      G1 left = xi * challenge + delta;
      G1 right = PcComputeCommitmentG(input.x_g_offset, sub_proof.z,
                                      sub_proof.z_delta);
      ret1 = left == right;
    };

    bool ret2 = false;
    tasks[1] = [&ret2, &com_pub, &com_ext_pub, &challenge, &sub_proof,
                &input]() {
      auto const& tau = com_pub.tau;
      auto const& beta = com_ext_pub.beta;
      G1 left = tau * challenge + beta;
      auto ip_za = InnerProduct(sub_proof.z, input.a);
      G1 right =
          PcComputeCommitmentG(input.y_g_offset, ip_za, sub_proof.z_beta);
      ret2 = left == right;
    };
    parallel::Invoke(tasks, true);

    assert(ret1 && ret2);
    return ret1 && ret2;
  }

  static void ComputeCom(CommitmentPub& com_pub, CommitmentSec const& com_sec,
                         ProverInput const& input) {
    // Tick tick(__FUNCTION__);
    std::array<parallel::Task, 2> tasks;
    tasks[0] = [&com_pub, &input, &com_sec]() {
      com_pub.xi =
          PcComputeCommitmentG(input.x_g_offset, input.x, com_sec.r_xi);
    };
    tasks[1] = [&com_pub, &input, &com_sec]() {
      com_pub.tau =
          PcComputeCommitmentG(input.y_g_offset, input.y, com_sec.r_tau);
    };
    parallel::Invoke(tasks, true);
  }

  // com(n) + com(1) + ip(n)
  static void ComputeCommitmentExt(CommitmentExtPub& com_ext_pub,
                                   CommitmentExtSec& com_ext_sec,
                                   ProverInput const& input) {
    // Tick tick(__FUNCTION__);
    auto n = input.n();
    com_ext_sec.d.resize(n);
    FrRand(com_ext_sec.d.data(), n);
    com_ext_sec.r_beta = FrRand();
    com_ext_sec.r_delta = FrRand();

    std::array<parallel::Task, 2> tasks;
    tasks[0] = [&com_ext_pub, &com_ext_sec, &input]() {
      com_ext_pub.delta = PcComputeCommitmentG(input.x_g_offset, com_ext_sec.d,
                                               com_ext_sec.r_delta);
    };
    tasks[1] = [&com_ext_pub, &input, &com_ext_sec]() {
      com_ext_pub.beta = PcComputeCommitmentG(
          input.y_g_offset, InnerProduct(input.a, com_ext_sec.d),
          com_ext_sec.r_beta);
    };
    parallel::Invoke(tasks, true);

    // std::cout << Tick::GetIndentString() << "multiexp(" << input.n() <<
    // ")\n";
  }

  static void UpdateSeed(h256_t& seed, CommitmentPub const& com_pub,
                         CommitmentExtPub const& com_ext_pub) {
    CryptoPP::Keccak_256 hash;
    HashUpdate(hash, seed);
    HashUpdate(hash, com_pub.xi);
    HashUpdate(hash, com_pub.tau);
    HashUpdate(hash, com_ext_pub.beta);
    HashUpdate(hash, com_ext_pub.delta);
    hash.Final(seed.data());
  }

  static void ComputeSubProof(SubProof& sub_proof, ProverInput const& input,
                              CommitmentSec const& com_sec,
                              CommitmentExtSec const& com_ext_sec,
                              Fr const& challenge) {
    // z = c * x + d
    sub_proof.z = input.x * challenge + com_ext_sec.d;
    sub_proof.z_delta = challenge * com_sec.r_xi + com_ext_sec.r_delta;
    sub_proof.z_beta = challenge * com_sec.r_tau + com_ext_sec.r_beta;
  }

  static void Prove(Proof& proof, h256_t seed, ProverInput const& input,
                    CommitmentPub com_pub, CommitmentSec com_sec) {
    Tick tick(__FUNCTION__);

    assert(PcBase::kGSize >= input.n());

    CommitmentExtSec com_ext_sec;
    ComputeCommitmentExt(proof.com_ext_pub, com_ext_sec, input);

    UpdateSeed(seed, com_pub, proof.com_ext_pub);
    Fr challenge = H256ToFr(seed);

    ComputeSubProof(proof.sub_proof, input, com_sec, com_ext_sec, challenge);
  }

  static bool Verify(Proof const& proof, h256_t seed,
                     VerifierInput const& input) {
    // Tick tick(__FUNCTION__);
    assert(PcBase::kGSize >= proof.n());
    if (input.a.size() != proof.sub_proof.z.size() || input.a.empty())
      return false;

    UpdateSeed(seed, input.com_pub, proof.com_ext_pub);
    Fr challenge = H256ToFr(seed);

    return VerifyInternal(input, challenge, proof.com_ext_pub, proof.sub_proof);
  }

  static bool Test(int64_t n);
};

// save to bin
template <typename Ar>
void serialize(Ar& ar, A2::CommitmentPub const& t) {
  ar& YAS_OBJECT_NVP("a2.cp", ("xi", t.xi), ("tau", t.tau));
}

// load from bin
template <typename Ar>
void serialize(Ar& ar, A2::CommitmentPub& t) {
  ar& YAS_OBJECT_NVP("a2.cp", ("xi", t.xi), ("tau", t.tau));
}

// save to bin
template <typename Ar>
void serialize(Ar& ar, A2::CommitmentExtPub const& t) {
  ar& YAS_OBJECT_NVP("a2.cep", ("delta", t.delta), ("beta", t.beta));
}

// load from bin
template <typename Ar>
void serialize(Ar& ar, A2::CommitmentExtPub& t) {
  ar& YAS_OBJECT_NVP("a2.cep", ("delta", t.delta), ("beta", t.beta));
}

// save to bin
template <typename Ar>
void serialize(Ar& ar, A2::SubProof const& t) {
  ar& YAS_OBJECT_NVP("a2.sp", ("z", t.z), ("z_delta", t.z_delta),
                     ("z_beta", t.z_beta));
}

// load from bin
template <typename Ar>
void serialize(Ar& ar, A2::SubProof& t) {
  ar& YAS_OBJECT_NVP("a2.sp", ("z", t.z), ("z_delta", t.z_delta),
                     ("z_beta", t.z_beta));
}

// save to bin
template <typename Ar>
void serialize(Ar& ar, A2::Proof const& t) {
  ar& YAS_OBJECT_NVP("a2.pf", ("c", t.com_ext_pub), ("p", t.sub_proof));
}

// load from bin
template <typename Ar>
void serialize(Ar& ar, A2::Proof& t) {
  ar& YAS_OBJECT_NVP("a2.pf", ("c", t.com_ext_pub), ("p", t.sub_proof));
}

bool A2::Test(int64_t n) {
  Tick tick(__FUNCTION__);
  std::cout << "n = " << n << "\n";
  std::vector<Fr> x(n);
  FrRand(x.data(), n);
  std::vector<Fr> a(n);
  FrRand(a.data(), n);

  h256_t UpdateSeed = misc::RandH256();

  int64_t x_g_offset = 30;
  int64_t y_g_offset = -1;
  auto z = InnerProduct(x, a);
  ProverInput prover_input(x, a, z, x_g_offset, y_g_offset);

  CommitmentPub com_pub;
  CommitmentSec com_sec(FrRand(), FrRand());
  ComputeCom(com_pub, com_sec, prover_input);

  Proof rom_proof;
  Prove(rom_proof, UpdateSeed, prover_input, com_pub, com_sec);

  VerifierInput verifier_input(a, com_pub, x_g_offset, y_g_offset);
  bool success = Verify(rom_proof, UpdateSeed, verifier_input);
  std::cout << __FILE__ << " " << __FUNCTION__ << ": " << success << "\n";
  return success;
}
}  // namespace hyrax
