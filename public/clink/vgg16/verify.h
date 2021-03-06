#pragma once

#include "./adapt.h"
#include "./conv_verify.h"
#include "./dense_verify.h"
#include "./image_com.h"
#include "./pooling_verify.h"
#include "./prove.h"
#include "./relu_verify.h"

namespace clink::vgg16 {

inline bool Verify(h256_t seed, std::string const& pub_path,
                   dbl::Image const& test_image, Proof const& proof) {
  Tick tick(__FN__);
  VerifyContext context(pub_path);
  AdaptVerifyItemMan adapt_man;
  R1csVerifyItemMan r1cs_man;

  std::vector<parallel::BoolTask> tasks;

  // check test image
  tasks.emplace_back([&context, &test_image]() {
    Image image(test_image);
    auto c = pc::ComputeCom(image.data, FrZero());
    return c == context.image_com_pub().c[0];
  });

  // conv
#if 0
  for (size_t i = 0; i < kConvCount; ++i) {
    tasks.emplace_back([&context, &seed, &proof, i, &adapt_man, &r1cs_man]() {
      return OneConvVerifyPreprocess(seed, context, kConvLayers[i],
                                     proof.conv[i], adapt_man, r1cs_man);
    });
  }
#endif

  for (size_t i = 0; i < kConvCount; ++i) {
    Tick tick_conv(__FN__, "conv" + std::to_string(i));
    std::unique_ptr<AdaptVerifyItemMan> padapt_man_conv(new AdaptVerifyItemMan);
    auto& adapt_man_conv = *padapt_man_conv;
    std::unique_ptr<R1csVerifyItemMan> pr1cs_man_conv(new R1csVerifyItemMan);
    auto& r1cs_man_conv = *pr1cs_man_conv;

    OneConvVerifyPreprocess(seed, context, kConvLayers[i], proof.conv[i],
                            adapt_man_conv, r1cs_man_conv);
    CHECK(AdaptVerify(seed, adapt_man_conv, proof.conv_adapt_proof[i]), "");
    CHECK(R1csVerify(seed, r1cs_man_conv, proof.conv_r1cs_proof[i]), "");
  }

#if 1
  // relubn
  tasks.emplace_back([&context, &seed, &proof, &adapt_man, &r1cs_man]() {
    return ReluVerifyPreprocess(seed, context, proof.relubn, adapt_man,
                                r1cs_man);
  });

  // pooling
  tasks.emplace_back([&context, &seed, &proof, &adapt_man, &r1cs_man]() {
    return PoolingVerifyPreprocess(seed, context, proof.pooling, adapt_man,
                                   r1cs_man);
  });

  // dense0
  tasks.emplace_back([&context, &seed, &proof]() {
    return DenseVerify<0>(seed, context, proof.dense0);
  });

  // dense1
  tasks.emplace_back([&context, &seed, &proof]() {
    return DenseVerify<1>(seed, context, proof.dense1);
  });

  // dense2
  tasks.emplace_back([&context, &seed, &proof]() {
    return DenseVerify<2>(seed, context, proof.dense2);
  });
#endif

  bool all_success = false;
  auto f1 = [&tasks](int64_t i) { return tasks[i](); };
  parallel::For(&all_success, tasks.size(), f1);
  CHECK(all_success, "verify failed");

  std::vector<parallel::BoolTask> bool_tasks;
  bool_tasks.emplace_back([&seed, &adapt_man, &proof]() {
    return AdaptVerify(seed, adapt_man, proof.adapt_proof);
  });

  bool_tasks.emplace_back([&seed, &r1cs_man, &proof]() {
    return R1csVerify(seed, r1cs_man, proof.r1cs_proof);
  });

  auto f2 = [&bool_tasks](int64_t i) { return bool_tasks[i](); };

  parallel::For(&all_success, bool_tasks.size(), f2);

  CHECK(all_success, "verify failed");

  return all_success;
}
}  // namespace clink::vgg16
