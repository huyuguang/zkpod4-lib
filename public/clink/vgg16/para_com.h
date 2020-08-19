#pragma once

#include "./auxi_pub.h"
#include "./para_fr.h"
#include "ecc/ecc.h"
#include "ecc/pc_base.h"
#include "misc/misc.h"
#include "public.h"
#include "utils/fst.h"

namespace clink::vgg16 {

struct ConvCommitmentPub {
  std::array<std::array<G1, 9>, kConvCount> coef;
  std::array<G1, kConvCount> bias;

  bool operator==(ConvCommitmentPub const& b) const {
    return coef == b.coef && bias == b.bias;
  }

  bool operator!=(ConvCommitmentPub const& b) const { return !(*this == b); }

  template <typename Ar>
  void serialize(Ar& ar) const {
    ar& YAS_OBJECT_NVP("vgg16.para.compub.conv", ("c", coef), ("b", bias));
  }

  template <typename Ar>
  void serialize(Ar& ar) {
    ar& YAS_OBJECT_NVP("vgg16.para.compub.conv", ("c", coef), ("b", bias));
  }
};

struct ConvCommitmentSec {
  std::array<std::array<Fr, 9>, kConvCount> coef_r;
  std::array<Fr, kConvCount> bias_r;

  bool operator==(ConvCommitmentSec const& b) const {
    return coef_r == b.coef_r && bias_r == b.bias_r;
  }

  bool operator!=(ConvCommitmentSec const& b) const { return !(*this == b); }

  template <typename Ar>
  void serialize(Ar& ar) const {
    ar& YAS_OBJECT_NVP("vgg16.para.comsec.conv", ("c", coef_r), ("b", bias_r));
  }

  template <typename Ar>
  void serialize(Ar& ar) {
    ar& YAS_OBJECT_NVP("vgg16.para.comsec.conv", ("c", coef_r), ("b", bias_r));
  }
};

struct DenseCommitmentPub {
  std::array<G1, 4096> d0;
  std::array<G1, 4096> d1;
  std::array<G1, 1000> d2;

  template <size_t Order>
  auto const& get() const {
    static_assert(Order == 0 || Order == 1 || Order == 2, "invalid Order");
    if constexpr (Order == 0)
      return d0;
    else if constexpr (Order == 1)
      return d1;
    else
      return d2;
  }

  bool operator==(DenseCommitmentPub const& b) const {
    return d0 == b.d0 && d1 == b.d1 && d2 == b.d2;
  }

  bool operator!=(DenseCommitmentPub const& b) const { return !(*this == b); }

  template <typename Ar>
  void serialize(Ar& ar) const {
    ar& YAS_OBJECT_NVP("vgg16.para.compub.dense", ("d0", d0), ("d1", d1),
                       ("d2", d2));
  }

  template <typename Ar>
  void serialize(Ar& ar) {
    ar& YAS_OBJECT_NVP("vgg16.para.compub.dense", ("d0", d0), ("d1", d1),
                       ("d2", d2));
  }
};

struct DenseCommitmentSec {
  std::array<Fr, 4096> d0_r;
  std::array<Fr, 4096> d1_r;
  std::array<Fr, 1000> d2_r;

  template <size_t Order>
  auto const& get() const {
    static_assert(Order == 0 || Order == 1 || Order == 2, "invalid Order");
    if constexpr (Order == 0)
      return d0_r;
    else if constexpr (Order == 1)
      return d1_r;
    else
      return d2_r;
  }

  bool operator==(DenseCommitmentSec const& b) const {
    return d0_r == b.d0_r && d1_r == b.d1_r && d2_r == b.d2_r;
  }

  bool operator!=(DenseCommitmentSec const& b) const { return !(*this == b); }

  template <typename Ar>
  void serialize(Ar& ar) const {
    ar& YAS_OBJECT_NVP("vgg16.para.comsec.dense", ("d0", d0_r), ("d1", d1_r),
                       ("d2", d2_r));
  }

  template <typename Ar>
  void serialize(Ar& ar) {
    ar& YAS_OBJECT_NVP("vgg16.para.comsec.dense", ("d0", d0_r), ("d1", d1_r),
                       ("d2", d2_r));
  }
};

struct ParaCommitmentPub {
  ConvCommitmentPub conv;
  DenseCommitmentPub dense;

  ParaCommitmentPub() {}

  ParaCommitmentPub(std::string const& file) {
    CHECK(YasLoadBin(file, *this), file);
  }

  bool operator==(ParaCommitmentPub const& b) const {
    return conv == b.conv && dense == b.dense;
  }

  bool operator!=(ParaCommitmentPub const& b) const { return !(*this == b); }

  template <typename Ar>
  void serialize(Ar& ar) const {
    ar& YAS_OBJECT_NVP("vgg16.para.compub", ("c", conv), ("d", dense));
  }
  template <typename Ar>
  void serialize(Ar& ar) {
    ar& YAS_OBJECT_NVP("vgg16.para.compub", ("c", conv), ("d", dense));
  }
};

struct ParaCommitmentSec {
  ConvCommitmentSec conv;
  DenseCommitmentSec dense;

  ParaCommitmentSec() {}

  ParaCommitmentSec(std::string const& file) {
    CHECK(YasLoadBin(file, *this), file);
  }

  bool operator==(ParaCommitmentSec const& b) const {
    return conv == b.conv && dense == b.dense;
  }

  bool operator!=(ParaCommitmentSec const& b) const { return !(*this == b); }

  template <typename Ar>
  void serialize(Ar& ar) const {
    ar& YAS_OBJECT_NVP("vgg16.para.comsec", ("c", conv), ("d", dense));
  }
  template <typename Ar>
  void serialize(Ar& ar) {
    ar& YAS_OBJECT_NVP("vgg16.para.comsec", ("c", conv), ("d", dense));
  }
};

inline void ComputeConvCommitment(
    std::array<Para::ConvLayer, kConvCount> const& para, AuxiPub const& auxi,
    ConvCommitmentPub& pub, ConvCommitmentSec& sec) {
  Tick tick(__FN__);
  auto parallel_f = [&para, &auxi, &pub, &sec](int64_t o) {
    auto C = kConvLayerInfos[o].C;
    auto K = kConvLayerInfos[o].K;

    auto const& layer = para[o];
    auto& pub_coef = pub.coef[o];
    auto& pub_bias = pub.bias[o];
    auto& sec_coef = sec.coef_r[o];
    auto& sec_bias = sec.bias_r[o];

    FrRand(sec_coef.data(), sec_coef.size());
    sec_bias = FrRand();

    auto get_coef_u = [&auxi, o](int64_t i) -> G1 const& {
      auto range = auxi.para_u_conv_coef(o);
      return i ? range.first[i - 1] : pc::PcH();
    };

    auto parallel_c = [&pub_coef, &get_coef_u, &layer, &sec_coef, K,
                       C](int64_t j) {
      auto const& coefs = layer.coefs;
      auto const& coefs_r = sec_coef[j];
      auto get_coef = [&coefs, &coefs_r, j, C](int64_t i) -> Fr const& {
        return i ? coefs[(i - 1) / C][(i - 1) % C][j / 3][j % 3] : coefs_r;
      };
      pub_coef[j] = MultiExpBdlo12<G1>(get_coef_u, get_coef, K * C + 1);
    };
    parallel::For(9, parallel_c);

    auto get_bias_u = [&auxi, o](int64_t i) -> G1 const& {
      auto range = auxi.para_u_conv_bias(o);
      return i ? range.first[i - 1] : pc::PcH();
    };

    auto const& bias = layer.bias;
    auto get_bias = [&bias, &sec_bias](int64_t i) -> Fr const& {
      return i ? bias[i - 1] : sec_bias;
    };
    pub_bias = MultiExpBdlo12<G1>(get_bias_u, get_bias, K + 1);
  };

  parallel::For((int64_t)para.size(), parallel_f);
}

inline void ComputeDenseCommitment(
    std::array<Para::DenseLayer, kDenseCount> const& para,
    DenseCommitmentPub& pub, DenseCommitmentSec& sec) {
  Tick tick(__FN__);
  FrRand(sec.d0_r.data(), sec.d0_r.size());
  FrRand(sec.d1_r.data(), sec.d1_r.size());
  FrRand(sec.d2_r.data(), sec.d2_r.size());

  CHECK(para[0].weight.size() == pub.d0.size(), "");
  CHECK(para[1].weight.size() == pub.d1.size(), "");
  CHECK(para[2].weight.size() == pub.d2.size(), "");

  auto pf1 = [&para, &pub, &sec](int64_t i) {
    auto const& w = para[0].weight[i];
    pub.d0[i] = pc::ComputeCom(w, sec.d0_r[i]);
  };
  parallel::For(para[0].weight.size(), pf1);

  auto pf2 = [&para, &pub, &sec](int64_t i) {
    auto const& w = para[1].weight[i];
    pub.d1[i] = pc::ComputeCom(w, sec.d1_r[i]);
  };
  parallel::For(para[1].weight.size(), pf2);

  auto pf3 = [&para, &pub, &sec](int64_t i) {
    auto const& w = para[2].weight[i];
    pub.d2[i] = pc::ComputeCom(w, sec.d2_r[i]);
  };
  parallel::For(para[2].weight.size(), pf3);
}

inline void ComputeParaCommitment(Para const& para, AuxiPub const& auxi,
                                  ParaCommitmentPub& pub,
                                  ParaCommitmentSec& sec) {
  Tick tick(__FN__);
  std::array<parallel::VoidTask, 2> tasks;
  tasks[0] = [&para, &auxi, &pub, &sec]() {
    ComputeConvCommitment(para.conv_layers(), auxi, pub.conv, sec.conv);
  };
  tasks[1] = [&para, &auxi, &pub, &sec]() {
    ComputeDenseCommitment(para.dense_layers(), pub.dense, sec.dense);
  };
  parallel::Invoke(tasks);
}
}  // namespace clink::vgg16