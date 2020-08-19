#pragma once

#include "./para_fr.h"
#include "ecc/ecc.h"
#include "ecc/pc_base.h"
#include "misc/misc.h"
#include "public.h"
#include "utils/fst.h"

namespace clink::vgg16 {

class AuxiPub {
 public:
  AuxiPub() {
    Tick tick(__FN__);
    InitPtr();

    ComputeParaConv<224, 64, 64>(*para_u_conv1_);
    ComputeParaConv<112, 128, 128>(*para_u_conv3_);
    ComputeParaConv<56, 256, 256>(*para_u_conv6_);
    ComputeParaConv<28, 512, 512>(*para_u_conv9_);
    ComputeParaConv<14, 512, 512>(*para_u_conv12_);

    ComputeDataConv<224, 3, 64>(*data_u_conv0_);
    ComputeDataConv<224, 64, 64>(*data_u_conv1_);
    ComputeDataConv<112, 64, 128>(*data_u_conv2_);
    ComputeDataConv<112, 128, 128>(*data_u_conv3_);
    ComputeDataConv<56, 128, 256>(*data_u_conv4_);
    ComputeDataConv<56, 256, 256>(*data_u_conv6_);
    ComputeDataConv<28, 256, 512>(*data_u_conv7_);
    ComputeDataConv<28, 512, 512>(*data_u_conv9_);
    ComputeDataConv<14, 512, 512>(*data_u_conv12_);
  }

  AuxiPub(std::string const& file) {
    Tick tick(__FN__);
    InitPtr();

    CHECK(YasLoadBin(file, *this), file);
  }

  bool operator==(AuxiPub const& b) const {
    return *para_u_conv1_ == *b.para_u_conv1_ &&
           *para_u_conv3_ == *b.para_u_conv3_ &&
           *para_u_conv6_ == *b.para_u_conv6_ &&
           *para_u_conv9_ == *b.para_u_conv9_ &&
           *para_u_conv12_ == *b.para_u_conv12_ &&
           *data_u_conv0_ == *b.data_u_conv0_ &&
           *data_u_conv1_ == *b.data_u_conv1_ &&
           *data_u_conv2_ == *b.data_u_conv2_ &&
           *data_u_conv3_ == *b.data_u_conv3_ &&
           *data_u_conv4_ == *b.data_u_conv4_ &&
           *data_u_conv6_ == *b.data_u_conv6_ &&
           *data_u_conv7_ == *b.data_u_conv7_ &&
           *data_u_conv9_ == *b.data_u_conv9_ &&
           *data_u_conv12_ == *b.data_u_conv12_;
  }

  bool operator!=(AuxiPub const& b) const { return !(*this == b); }

  template <typename Ar>
  void serialize(Ar& ar) const {
    ar& YAS_OBJECT_NVP(
        "vgg16.auxi", ("c1", *para_u_conv1_), ("c3", *para_u_conv3_),
        ("c6", *para_u_conv6_), ("c9", *para_u_conv9_),
        ("c12", *para_u_conv12_), ("d0", *data_u_conv0_),
        ("d1", *data_u_conv1_), ("d2", *data_u_conv2_), ("d3", *data_u_conv3_),
        ("d4", *data_u_conv4_), ("d6", *data_u_conv6_), ("d7", *data_u_conv7_),
        ("d9", *data_u_conv9_), ("d12", *data_u_conv12_));
  }

  template <typename Ar>
  void serialize(Ar& ar) {
    ar& YAS_OBJECT_NVP(
        "vgg16.auxi", ("c1", *para_u_conv1_), ("c3", *para_u_conv3_),
        ("c6", *para_u_conv6_), ("c9", *para_u_conv9_),
        ("c12", *para_u_conv12_), ("d0", *data_u_conv0_),
        ("d1", *data_u_conv1_), ("d2", *data_u_conv2_), ("d3", *data_u_conv3_),
        ("d4", *data_u_conv4_), ("d6", *data_u_conv6_), ("d7", *data_u_conv7_),
        ("d9", *data_u_conv9_), ("d12", *data_u_conv12_));
  }

  // para conv coef
  std::pair<G1 const*, G1 const*> para_u_conv_coef(size_t order) const {
    return std::make_pair(
        para_u_conv_ptr_[order],
        para_u_conv_ptr_[order] + para_u_conv_coef_size_[order]);
  }

  // para conv bias
  std::pair<G1 const*, G1 const*> para_u_conv_bias(size_t order) const {
    return std::make_pair(
        para_u_conv_ptr_[order],
        para_u_conv_ptr_[order] + para_u_conv_bias_size_[order]);
  }

  // data conv
  std::pair<G1 const*, G1 const*> data_u_conv(size_t order) const {
    return std::make_pair(data_u_conv_ptr_[order],
                          data_u_conv_ptr_[order] + data_u_conv_size_[order]);
  }

 private:
  // $u_i=\prod_{j=0}^{DD-1}g_{iDD+j},i\in[0,KC-1]$
  std::unique_ptr<std::array<G1, 64 * 64>> para_u_conv1_;
  std::unique_ptr<std::array<G1, 128 * 128>> para_u_conv3_;
  std::unique_ptr<std::array<G1, 256 * 256>> para_u_conv6_;
  std::unique_ptr<std::array<G1, 512 * 512>> para_u_conv9_;
  std::unique_ptr<std::array<G1, 512 * 512>> para_u_conv12_;
  std::array<G1 const*, kConvCount> para_u_conv_ptr_;
  std::array<size_t, kConvCount> para_u_conv_coef_size_;
  std::array<size_t, kConvCount> para_u_conv_bias_size_;

  // $u_i=\prod_{j=0}^{K-1}g_{i+jCDD},i\in[0,CDD-1]$
  std::unique_ptr<std::array<G1, 3 * 224 * 224>> data_u_conv0_;
  std::unique_ptr<std::array<G1, 64 * 224 * 224>> data_u_conv1_;
  std::unique_ptr<std::array<G1, 64 * 112 * 112>> data_u_conv2_;
  std::unique_ptr<std::array<G1, 128 * 112 * 112>> data_u_conv3_;
  std::unique_ptr<std::array<G1, 128 * 56 * 56>> data_u_conv4_;
  std::unique_ptr<std::array<G1, 256 * 56 * 56>> data_u_conv6_;
  std::unique_ptr<std::array<G1, 256 * 28 * 28>> data_u_conv7_;
  std::unique_ptr<std::array<G1, 512 * 28 * 28>> data_u_conv9_;
  std::unique_ptr<std::array<G1, 512 * 14 * 14>> data_u_conv12_;
  std::array<G1 const*, kConvCount> data_u_conv_ptr_;
  std::array<size_t, kConvCount> data_u_conv_size_;

 private:
  void InitPtr() {
    para_u_conv1_.reset(new std::array<G1, 64 * 64>);
    para_u_conv3_.reset(new std::array<G1, 128 * 128>);
    para_u_conv6_.reset(new std::array<G1, 256 * 256>);
    para_u_conv9_.reset(new std::array<G1, 512 * 512>);
    para_u_conv12_.reset(new std::array<G1, 512 * 512>);

    // C * D * D
    data_u_conv0_.reset(new std::array<G1, 3 * 224 * 224>);
    data_u_conv1_.reset(new std::array<G1, 64 * 224 * 224>);
    data_u_conv2_.reset(new std::array<G1, 64 * 112 * 112>);
    data_u_conv3_.reset(new std::array<G1, 128 * 112 * 112>);
    data_u_conv4_.reset(new std::array<G1, 128 * 56 * 56>);
    data_u_conv6_.reset(new std::array<G1, 256 * 56 * 56>);
    data_u_conv7_.reset(new std::array<G1, 256 * 28 * 28>);
    data_u_conv9_.reset(new std::array<G1, 512 * 28 * 28>);
    data_u_conv12_.reset(new std::array<G1, 512 * 14 * 14>);

    para_u_conv_ptr_ = std::array<G1 const*, kConvCount>{
        {para_u_conv1_->data(), para_u_conv1_->data(), para_u_conv3_->data(),
         para_u_conv3_->data(), para_u_conv6_->data(), para_u_conv6_->data(),
         para_u_conv6_->data(), para_u_conv9_->data(), para_u_conv9_->data(),
         para_u_conv9_->data(), para_u_conv12_->data(), para_u_conv12_->data(),
         para_u_conv12_->data()}};
    para_u_conv_coef_size_ = std::array<size_t, kConvCount>{
        {192, 4096, 8192, 16384, 32768, 65536, 65536, 131072, 262144, 262144,
         262144, 262144, 262144}};
    para_u_conv_bias_size_ = std::array<size_t, kConvCount>{
        {64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512}};

    data_u_conv_ptr_ = std::array<G1 const*, kConvCount>{
        {data_u_conv0_->data(), data_u_conv1_->data(), data_u_conv2_->data(),
         data_u_conv3_->data(), data_u_conv4_->data(), data_u_conv6_->data(),
         data_u_conv6_->data(), data_u_conv7_->data(), data_u_conv9_->data(),
         data_u_conv9_->data(), data_u_conv12_->data(), data_u_conv12_->data(),
         data_u_conv12_->data()}};

    data_u_conv_size_ = std::array<size_t, kConvCount>{
        {data_u_conv0_->size(), data_u_conv1_->size(), data_u_conv2_->size(),
         data_u_conv3_->size(), data_u_conv4_->size(), data_u_conv6_->size(),
         data_u_conv6_->size(), data_u_conv7_->size(), data_u_conv9_->size(),
         data_u_conv9_->size(), data_u_conv12_->size(), data_u_conv12_->size(),
         data_u_conv12_->size()}};
  }

  template <size_t D, size_t C, size_t K>
  void ComputeParaConv(std::array<G1, K * C>& u) {
    auto DD = D * D;
    auto KC = K * C;
    auto pf = [&u, DD](int64_t i) { u[i] = pc::ComputeSigmaG(i * DD, DD); };
    parallel::For(KC, pf);
  }

  template <size_t D, size_t C, size_t K>
  void ComputeDataConv(std::array<G1, C * D * D>& u) {
    auto CDD = C * D * D;
    auto pf = [&u, CDD](int64_t i) {
      u[i] = G1Zero();
      for (size_t j = 0; j < K; ++j) {
        u[i] += pc::PcG(i + j * CDD);
      }
    };
    parallel::For(CDD, pf);
    // for (size_t i = 0; i < C * D * D; ++i) {
    //  u[i] = G1Zero();
    //  for (size_t j = 0; j < K; ++j) {
    //    u[i] += pc::PcG(i + j * C * D * D);
    //  }
    //}
  }

  bool Load(std::string const& file) {
    try {
      yas::file_istream is(file.c_str());
      yas::binary_iarchive<yas::file_istream, YasBinF()> ia(is);
      ia.serialize(*this);
      return true;
    } catch (std::exception& e) {
      std::cerr << e.what() << "\n";
      return false;
    }
  }
};
};  // namespace clink::vgg16