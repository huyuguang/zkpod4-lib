#pragma once

#include "./match.h"
#include "./pack.h"

namespace pc_utils::matchpack {

struct Proof {
  match::Proof match_proof;
  pack::Proof pack_proof;
  G1 com_pack_y;
};

inline bool operator==(Proof const& a, Proof const& b) {
  return a.match_proof == b.match_proof && a.pack_proof == b.pack_proof &&
         a.com_pack_y == b.com_pack_y;
}

inline bool operator!=(Proof const& a, Proof const& b) { return !(a == b); }

// save to bin
template <typename Ar>
void serialize(Ar& ar, Proof const& t) {
  ar& YAS_OBJECT_NVP("mp.p", ("s", t.match_proof), ("p", t.pack_proof),
                     ("y", t.com_pack_y));
}

// load from bin
template <typename Ar>
void serialize(Ar& ar, Proof& t) {
  ar& YAS_OBJECT_NVP("mp.p", ("s", t.match_proof), ("p", t.pack_proof),
                     ("y", t.com_pack_y));
}

struct ProveOutput {
  match::Proof match_proof;
  pack::Proof pack_proof;
  std::vector<Fr> y;
  Fr com_y_r;
  G1 com_pack_y;
  std::vector<Fr> pack_y;
  Fr com_pack_y_r;
  Proof BuildProof() const {
    Proof ret;
    ret.match_proof = match_proof;
    ret.pack_proof = pack_proof;
    ret.com_pack_y = com_pack_y;
    return ret;
  }
};

struct ProverInput {
  ProverInput(Fr const& k, std::vector<Fr> const& x, G1 const& com_x,
              Fr const& com_x_r, int64_t x_g_offset, int64_t py_g_offset)
      : k(k),
        x(x),
        com_x(com_x),
        com_x_r(com_x_r),
        n((int64_t)x.size()),
        x_g_offset(x_g_offset),
        py_g_offset(py_g_offset) {
#ifdef _DEBUG
    assert(com_x == PcComputeCommitmentG(x_g_offset, x, com_x_r, true));
#endif
  }
  Fr const& k;
  std::vector<Fr> const& x;
  G1 const& com_x;
  Fr const& com_x_r;
  int64_t const n;
  int64_t const x_g_offset;
  int64_t const py_g_offset;
};

inline void Prove(ProveOutput& output, h256_t seed, ProverInput const& input) {
  // prove y[i]=x[i] == k, i=[0,s)
  int64_t n = input.n;
  output.y.resize(n);
  for (int64_t i = 0; i < n; ++i) {
    output.y[i] = input.x[i] == input.k ? 1 : 0;
  }

  output.com_y_r = FrRand();
  auto com_y =
      PcComputeCommitmentG(input.x_g_offset, output.y, output.com_y_r, true);

  match::ProverInput m_input(input.k, input.x, input.com_x, input.com_x_r,
                             output.y, com_y, output.com_y_r, input.x_g_offset);
  auto& match_proof = output.match_proof;
  match::Prove(match_proof, seed, m_input);
  assert(match_proof.com_w.back() == com_y);

  // pack y to pack_y
  output.pack_y = FrBitsToFrs(output.y);
  output.com_pack_y_r = FrRand();
  output.com_pack_y = PcComputeCommitmentG(input.py_g_offset, output.pack_y,
                                           output.com_pack_y_r);
  auto& pack_proof = output.pack_proof;
  pack::ProverInput p_input(output.y, com_y, output.com_y_r, input.x_g_offset,
                            output.pack_y, output.com_pack_y,
                            output.com_pack_y_r, input.py_g_offset);
  pack::Prove(pack_proof, seed, p_input);
}

struct VerifierInput {
  VerifierInput(int64_t n, Fr const& k, int64_t const x_g_offset,
                int64_t const py_g_offset)
      : n(n), k(k), x_g_offset(x_g_offset), py_g_offset(py_g_offset) {}
  int64_t n;
  Fr const& k;
  int64_t const x_g_offset;
  int64_t const py_g_offset;
};

inline bool Verify(Proof const& proof, h256_t seed,
                   VerifierInput const& input) {
  match::VerifierInput m_input(input.n, input.k, input.x_g_offset);
  if (!match::Verify(proof.match_proof, seed, m_input)) return false;

  G1 const& com_y = proof.match_proof.com_w.back();
  pack::VerifierInput p_input(input.n, com_y, input.x_g_offset,
                              proof.com_pack_y, input.py_g_offset);
  return pack::Verify(proof.pack_proof, seed, p_input);
}

inline bool Test() {
  auto seed = misc::RandH256();
  int64_t x_g_offset = 540;
  int64_t py_g_offset =20;
  int64_t n = 1000;
  Fr k = FrRand();
  std::vector<Fr> x(n);
  FrRand(x);
  x[rand() % n] = k;
  auto com_x_r = FrRand();
  auto com_x = PcComputeCommitmentG(x_g_offset, x, com_x_r);

  ProverInput prover_input(k, x, com_x, com_x_r, x_g_offset, py_g_offset);
  ProveOutput output;
  Prove(output, seed, prover_input);

  auto proof = output.BuildProof();
  if (proof.match_proof.com_x() != com_x) {
    assert(false);
    return false;
  }

  VerifierInput verifier_input(n, k, x_g_offset, py_g_offset);
  bool success = Verify(proof, seed, verifier_input);
  std::cout << __FILE__ << " " << __FUNCTION__ << ": " << success << "\n";
  return success;
}

}  // namespace pc_utils::matchpack