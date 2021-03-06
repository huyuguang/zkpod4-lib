#pragma once

#include "./details.h"
#include "./equal_ip.h"

// x: vector<Fr>, (size+252)/253 = n, {0,1}
// y: vector<Fr>, size = n
// open com(gx,x), com(gy,y)
// prove binary of y == x
// let b: vector<Fr>, size = n, b = fst(com(x), com(y))
// let c: vector<Fr>, size = 253, a = {2^0, 2^1.... 2^252}
// let a = {{c*b0}, {c*b1}, {c*b2}.... {c*b252}}, a.size = 253*n
// prove <x, a> == <y, b>

namespace clink {

template <typename HyraxA>
struct Pack {
  using Proof = typename EqualIp<HyraxA>::Proof;

  struct ProveInput {
    ProveInput(std::vector<Fr> const& x, G1 const& com_x, Fr const& com_x_r,
               GetRefG1 const& get_gx, std::vector<Fr> const& y,
               G1 const& com_y, Fr const& com_y_r, GetRefG1 const& get_gy)
        : x(x),
          com_x(com_x),
          com_x_r(com_x_r),
          get_gx(get_gx),
          y(y),
          com_y(com_y),
          com_y_r(com_y_r),
          get_gy(get_gy) {
#ifdef _DEBUG
      assert((xn() + 252) / 253 == yn());
      assert(FrBitsToFrs(x) == y);
      auto check_x = FrsToFrBits(y);
      assert(check_x.size() >= x.size());
      for (auto i = x.size(); i < check_x.size(); ++i) {
        assert(check_x[i] == 0);
      }
      check_x.resize(x.size());
      assert(check_x == x);
#endif
    }
    int64_t xn() const { return (int64_t)x.size(); }
    int64_t yn() const { return (int64_t)y.size(); }
    std::vector<Fr> const& x;
    G1 const& com_x;
    Fr const& com_x_r;
    GetRefG1 const& get_gx;
    std::vector<Fr> const& y;
    G1 const& com_y;
    Fr const& com_y_r;
    GetRefG1 const& get_gy;
  };

  static void Prove(Proof& proof, h256_t seed, ProveInput const& input) {
    int64_t xn = input.xn();
    int64_t yn = input.yn();

    UpdateSeed(seed, input.com_x, input.com_y, xn);

    // b
    std::vector<Fr> b(yn);
    ComputeFst(seed, "pack:b", b);

    // c
    std::vector<Fr> c(253);
    c[0] = FrOne();
    for (size_t i = 1; i < c.size(); ++i) {
      c[i] = c[i - 1] + c[i - 1];
    }

    // a
    std::vector<Fr> a;
    a.reserve(yn * 253);
    for (int64_t i = 0; i < yn; ++i) {
      auto cb = c * b[i];
      a.insert(a.end(), cb.begin(), cb.end());
    }
    a.resize(xn);

    auto z = InnerProduct(input.y, b);

    assert(z == InnerProduct(input.x, a));

    typename EqualIp<HyraxA>::ProveInput eip_input(
        input.x, a, input.com_x, input.com_x_r, input.get_gx, input.y, b,
        input.com_y, input.com_y_r, input.get_gy, z);
    EqualIp<HyraxA>::Prove(proof, seed, eip_input);
  }

  struct VerifyInput {
    VerifyInput(int64_t xn, G1 const& com_x, GetRefG1 const& get_gx,
                G1 const& com_y, GetRefG1 const& get_gy)
        : xn(xn),
          yn((xn + 252) / 253),
          com_x(com_x),
          get_gx(get_gx),
          com_y(com_y),
          get_gy(get_gy) {}
    int64_t const xn;
    int64_t const yn;
    G1 const& com_x;
    GetRefG1 const& get_gx;
    G1 const& com_y;
    GetRefG1 const& get_gy;
  };

  // NOET: n is x.size()
  static bool Verify(Proof const& proof, h256_t seed,
                     VerifyInput const& input) {
    int64_t xn = input.xn;
    int64_t yn = input.yn;

    UpdateSeed(seed, input.com_x, input.com_y, xn);

    // b
    std::vector<Fr> b(yn);
    ComputeFst(seed, "pack:b", b);

    // c
    std::vector<Fr> c(253);
    c[0] = FrOne();
    for (size_t i = 1; i < c.size(); ++i) {
      c[i] = c[i - 1] + c[i - 1];
    }

    // a
    std::vector<Fr> a;
    a.reserve(yn * 253);
    for (int64_t i = 0; i < yn; ++i) {
      auto cb = c * b[i];
      a.insert(a.end(), cb.begin(), cb.end());
    }
    a.resize(xn);

    typename EqualIp<HyraxA>::VerifyInput eip_input(
        a, input.com_x, input.get_gx, b, input.com_y, input.get_gy);
    return EqualIp<HyraxA>::Verify(seed, proof, eip_input);
  }

  static bool Test(int64_t xn);

 private:
  static void UpdateSeed(h256_t& seed, G1 const& c1, G1 const& c2, int64_t n) {
    CryptoPP::Keccak_256 hash;
    HashUpdate(hash, seed);
    HashUpdate(hash, c1);
    HashUpdate(hash, c2);
    HashUpdate(hash, n);
    hash.Final(seed.data());
  }
};

template <typename HyraxA>
bool Pack<HyraxA>::Test(int64_t xn) {
  int64_t x_g_offset = 2990;
  int64_t y_g_offset = 670;
  GetRefG1 get_gx = [x_g_offset](int64_t i) -> G1 const& {
    return pc::PcG()[x_g_offset + i];
  };
  GetRefG1 get_gy = [y_g_offset](int64_t i) -> G1 const& {
    return pc::PcG()[y_g_offset + i];
  };

  auto seed = misc::RandH256();
  std::vector<Fr> x(xn);
  for (auto& i : x) i = rand() % 2;
  auto y = FrBitsToFrs(x);
  auto com_x_r = FrRand();
  auto com_x = pc::ComputeCom(get_gx, x, com_x_r);
  auto com_y_r = FrRand();
  auto com_y = pc::ComputeCom(get_gy, y, com_y_r);

  ProveInput prove_input(x, com_x, com_x_r, get_gx, y, com_y, com_y_r, get_gy);
  Proof proof;
  Prove(proof, seed, prove_input);

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

  VerifyInput verify_input(xn, com_x, get_gx, com_y, get_gy);
  bool success = Verify(proof, seed, verify_input);
  std::cout << __FILE__ << " " << __FN__ << ": " << success << "\n\n\n\n\n\n";
  return success;
}
}  // namespace clink