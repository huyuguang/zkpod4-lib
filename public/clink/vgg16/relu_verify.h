#pragma once

#include "./relu_prove.h"

namespace clink::vgg16 {

inline bool ReluInOutVerifyPreprocess(h256_t seed,
                                        VerifyContext const& context,
                                        ReluProof const& proof,
                                        AdaptVerifyItemMan& item_man) {
  auto const& io_pub = proof.io_pub;

  if (io_pub.cx.size() != kReluLayers.size() * 2 + 2) {
    std::cout << __FN__ << ": " << __LINE__ << ": proof invalid\n";
    return false;
  }

  for (size_t order = 0; order < kReluLayers.size(); ++order) {
    auto layer = kReluLayers[order];
    if (context.image_com_pub().c[layer] != io_pub.cx[2 + order * 2]) {
      std::cout << __FN__ << ": " << __LINE__ << ": proof invalid\n";
      return false;
    }
    if (context.image_com_pub().c[layer + 1] != io_pub.cx[2 + order * 2 + 1]) {
      std::cout << __FN__ << ": " << __LINE__ << ": proof invalid\n";
      return false;
    }
  }

  std::vector<std::vector<Fr>> a(io_pub.cx.size());
  size_t total_size = 0;
  for (size_t order = 0; order < kReluLayers.size(); ++order) {
    auto layer = kReluLayers[order];
    auto const& info = kImageInfos[layer];
    auto size = info.C * info.D * info.D;
    a[2 + order * 2].resize(size);
    a[2 + order * 2 + 1].resize(size);
    total_size += size;
  }
  a[0].resize(total_size);
  a[1].resize(total_size);

  ReluBnUpdateSeed(seed, io_pub.cx[0], io_pub.cx[1]);
  a[0] = ReluBnComputeFst(seed, "relubn input", total_size);
  a[1] = ReluBnComputeFst(seed, "relubn output", total_size);

  auto it_i = a[0].begin();
  auto it_o = a[1].begin();
  for (size_t i = 1; i < a.size() / 2; ++i) {
    auto& a_i = a[2 * i];
    auto& a_o = a[2 * i + 1];
    auto size = a_i.size();
    std::copy(it_i, it_i + size, a_i.begin());
    std::copy(it_o, it_o + size, a_o.begin());
    it_i += size;
    it_o += size;
  }

  AdaptVerifyItem adapt_item_in;
  AdaptVerifyItem adapt_item_out;
  adapt_item_in.Init(io_pub.cx.size() / 2, ReluAdaptTag(true));
  adapt_item_out.Init(io_pub.cx.size() / 2, ReluAdaptTag(false));
  for (size_t j = 0; j < io_pub.cx.size() / 2; ++j) {
    adapt_item_in.a[j] = std::move(a[j * 2]);
    adapt_item_in.cx[j] = io_pub.cx[j * 2];
    if (j == 0) adapt_item_in.a[j] = -adapt_item_in.a[j];

    adapt_item_out.a[j] = std::move(a[j * 2 + 1]);
    adapt_item_out.cx[j] = io_pub.cx[j * 2 + 1];
    if (j == 0) adapt_item_out.a[j] = -adapt_item_out.a[j];
  }
  item_man.emplace(std::move(adapt_item_in));
  item_man.emplace(std::move(adapt_item_out));
  return true;
}

inline bool ReluR1csVerifyPreprocess(h256_t seed,
                                       VerifyContext const& /*context*/,
                                       ReluProof const& proof,
                                       R1csVerifyItemMan& r1cs_man) {
  Tick tick(__FN__);
  namespace fp = circuit::fp;
  (void)seed;
  if (proof.r1cs_pub.com_w[0] != proof.io_pub.cx[0]) {  // in
    std::cout << __FN__ << ": " << __LINE__ << ": proof invalid\n";
    return false;
  }

  if (proof.r1cs_pub.com_w[1] != proof.io_pub.cx[1]) {  // out
    std::cout << __FN__ << ": " << __LINE__ << ": proof invalid\n";
    return false;
  }

  libsnark::protoboard<Fr> pb;
  fp::Relu2Gadget<8, 24 * 2, 24> gadget(pb, "vgg16 relu gadget");
  int64_t const primary_input_size = 0;
  pb.set_input_sizes(primary_input_size);
  auto n = ReluGetCircuitCount();

  R1csVerifyItem item;
  item.public_w.reset(new std::vector<std::vector<Fr>>);
  item.r1cs_info.reset(new R1csInfo(pb));
  item.r1cs_input.reset(new R1cs::VerifyInput(
      n, *item.r1cs_info, ReluR1csTag(), proof.r1cs_pub.com_w, *item.public_w,
      pc::kGetRefG1));

  r1cs_man.emplace(std::move(item));
  return true;
}

inline bool ReluVerifyPreprocess(h256_t seed, VerifyContext const& context,
                                   ReluProof const& proof,
                                   AdaptVerifyItemMan& item_man,
                                   R1csVerifyItemMan& r1cs_man) {
  Tick tick(__FN__);

  std::array<parallel::VoidTask, 2> tasks;
  tasks[0] = [&seed, &context, &proof, &item_man]() {
    CHECK(ReluInOutVerifyPreprocess(seed, context, proof, item_man), "");
  };

  tasks[1] = [&seed, &context, &proof, &r1cs_man]() {
    CHECK(ReluR1csVerifyPreprocess(seed, context, proof, r1cs_man), "");
  };

  return true;
}
}  // namespace clink::vgg16