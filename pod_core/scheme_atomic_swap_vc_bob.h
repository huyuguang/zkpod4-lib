#pragma once

#include "basic_types.h"
#include "ecc.h"
#include "parallel.h"
#include "public.h"
#include "scheme_atomic_swap_vc_notary.h"
#include "scheme_atomic_swap_vc_protocol.h"
#include "scheme_misc.h"

namespace scheme::atomic_swap_vc {

template <typename BobData>
class Bob {
 public:
  Bob(std::shared_ptr<BobData> b, h256_t const& self_id, h256_t const& peer_id,
      std::vector<Range> demands);

 public:
  void GetRequest(Request& request);
  bool OnResponse(Response response, Receipt& receipt);
  bool OnSecret(Secret const& secret);
  bool SaveDecrypted(std::string const& file);

 private:
  void BuildMapping();
  bool CheckEncryptedM() const;
  bool CheckKVW() const;
  void DecryptM(std::vector<Fr> const& v);

 private:
  std::shared_ptr<BobData> b_;
  h256_t const self_id_;
  h256_t const peer_id_;
  uint64_t const n_;
  uint64_t const s_;
  std::vector<Range> const demands_;
  uint64_t demands_count_ = 0;
  h256_t alice_nonce_;
  h256_t bob_nonce_;

 private:
  std::vector<G1> k_;   // sizeof() = (count + 1) * s
  std::vector<Fr> vw_;  // sizeof() = s

 private:
  struct Mapping {
    uint64_t global_index;
  };
  std::vector<Mapping> mappings_;
  Fr sigma_vw_;

 private:
  h256_t seed2_;
  std::vector<Fr> w_;  // size() is count + 1, and the last is FrOne
  std::vector<Fr> decrypted_m_;
  std::vector<Fr> encrypted_m_;
  Receipt receipt_;
  std::vector<Fr> plain_;
};

template <typename BobData>
using BobPtr = std::shared_ptr<Bob<BobData>>;
}  // namespace scheme::atomic_swap_vc

#include "scheme_atomic_swap_vc_bob.inc"