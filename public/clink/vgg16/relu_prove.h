#pragma once

#include "./adapt.h"
#include "./r1cs.h"
#include "./relu_pub.h"

namespace clink::vgg16 {

inline void ReluInOutProvePreprocess(h256_t seed, ProveContext const& context,
                                     ReluProof& proof, ReluInOutSec& io_sec,
                                     AdaptProveItemMan& adapt_man) {
  Tick tick(__FN__);
  auto& io_pub = proof.io_pub;
  std::vector<ReluBnImage> images;
  ReluBuildImages(context, images);

  ReluBnUpdateSeed(seed, images[0].com_x, images[1].com_x);

  ReluBnBuildChallenge(seed, images);

  io_pub.cx.resize(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    io_pub.cx[i] = images[i].com_x;
  }

  io_sec.input = images[0].x;
  io_sec.output = images[1].x;
  io_sec.com_in_r = images[0].com_x_r;
  io_sec.com_out_r = images[1].com_x_r;

  AdaptProveItem adapt_item_in;
  AdaptProveItem adapt_item_out;
  adapt_item_in.Init(images.size() / 2, ReluAdaptTag(true));
  adapt_item_out.Init(images.size() / 2, ReluAdaptTag(false));
  for (size_t j = 0; j < images.size() / 2; ++j) {
    adapt_item_in.x[j] = std::move(images[j * 2].x);
    adapt_item_in.a[j] = std::move(images[j * 2].a);
    adapt_item_in.cx[j] = images[j * 2].com_x;
    adapt_item_in.rx[j] = images[j * 2].com_x_r;
    if (j == 0) adapt_item_in.a[j] = -adapt_item_in.a[j];

    adapt_item_out.x[j] = std::move(images[j * 2 + 1].x);
    adapt_item_out.a[j] = std::move(images[j * 2 + 1].a);
    adapt_item_out.cx[j] = images[j * 2 + 1].com_x;
    adapt_item_out.rx[j] = images[j * 2 + 1].com_x_r;
    if (j == 0) adapt_item_out.a[j] = -adapt_item_out.a[j];
  }
  adapt_man.emplace(std::move(adapt_item_in));
  adapt_man.emplace(std::move(adapt_item_out));
}

// prove y=<x,para>
inline void ReluR1csProvePreprocess(h256_t seed, ProveContext const& /*context*/,
                                    ReluInOutSec const& io_sec,
                                    ReluProof& proof,
                                    std::shared_ptr<ReluR1csSec> pr1cs_sec,
                                    R1csProveItemMan& r1cs_man) {
  Tick tick(__FN__);
  namespace fp = circuit::fp;
  (void)seed;
  auto const& io_pub = proof.io_pub;
  auto& r1cs_pub = proof.r1cs_pub;
  auto& r1cs_sec = *pr1cs_sec;

  libsnark::protoboard<Fr> pb;
  fp::Relu2Gadget<8, 24 * 2, 24> gadget(pb, "vgg16 relu gadget");
  int64_t const primary_input_size = 0;
  pb.set_input_sizes(primary_input_size);
  auto r1cs_ret_index = gadget.ret().index - 1;  // see protoboard<FieldT>::val
  r1cs_sec.r1cs_info.reset(new R1csInfo(pb));
  auto s = r1cs_sec.r1cs_info->num_variables;
  r1cs_sec.w.resize(s);
  auto n = io_sec.input.size();
  for (auto& i : r1cs_sec.w) i.resize(n);
  std::cout << Tick::GetIndentString() << r1cs_sec.r1cs_info->to_string()
            << ", repeat times: " << n << "\n";

  for (size_t j = 0; j < n; ++j) {
    gadget.Assign(io_sec.input[j]);
    assert(pb.is_satisfied());
    auto v = pb.full_variable_assignment();
    for (int64_t i = 0; i < s; ++i) {
      r1cs_sec.w[i][j] = v[i];
    }
    CHECK(r1cs_sec.w[r1cs_ret_index][j] == io_sec.output[j], "");
  }

  r1cs_pub.com_w.resize(s);
  r1cs_sec.com_w_r.resize(s);

  // std::cout << "compute conv com(witness)\n";
  assert(r1cs_ret_index == 1);
  r1cs_sec.com_w_r[0] = io_sec.com_in_r;
  r1cs_pub.com_w[0] = io_pub.cx[0];  // in

  r1cs_sec.com_w_r[1] = io_sec.com_out_r;
  r1cs_pub.com_w[1] = io_pub.cx[1];  // out

  auto parallel_f = [&r1cs_sec, &r1cs_pub](int64_t i) {
    std::vector<Fr> const& x = r1cs_sec.w[i + 2];
    Fr& r = r1cs_sec.com_w_r[i + 2];
    G1& c = r1cs_pub.com_w[i + 2];
    r = FrRand();
    c = pc::ComputeCom(x, r, true);
  };
  parallel::For<int64_t>(s - 2, parallel_f);

  R1csProveItem item;
  item.r1cs_sec = pr1cs_sec;  // save ref
  item.r1cs_input.reset(new R1cs::ProveInput(
      *pr1cs_sec->r1cs_info, ReluR1csTag(), std::move(pr1cs_sec->w),
      r1cs_pub.com_w, pr1cs_sec->com_w_r, pc::kGetRefG1));
  r1cs_man.emplace(std::move(item));
}

inline void ReluBnProvePreprocess(h256_t seed, ProveContext const& context,
                                  ReluProof& proof,
                                  AdaptProveItemMan& adapt_man,
                                  R1csProveItemMan& r1cs_man) {
  ReluInOutSec io_sec;
  ReluInOutProvePreprocess(seed, context, proof, io_sec, adapt_man);

  std::shared_ptr<ReluR1csSec> r1cs_sec(new ReluR1csSec);
  ReluR1csProvePreprocess(seed, context, io_sec, proof, r1cs_sec, r1cs_man);
}
};  // namespace clink::vgg16
