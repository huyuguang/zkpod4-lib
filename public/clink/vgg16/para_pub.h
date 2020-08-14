#pragma once

#include "./policy.h"

namespace clink::vgg16 {

struct ConvLayerInfo {
  size_t const order;
  size_t const D;
  size_t const C;
  size_t const K;
  size_t input_count() const { return D * D * C; }
  size_t output_count() const { return D * D * K; }
};

struct DenseLayerInfo {
  size_t const order;
  size_t const input_count;
  size_t const type_count;
};

struct ImageInfo {
  size_t size() const { return C * D * D; }
  size_t const order;
  size_t const D;
  size_t const C;
};

// order, D, C, K
inline constexpr std::array<ConvLayerInfo, 13> kConvLayerInfos{
    ConvLayerInfo{0, 224, 3, 64},    ConvLayerInfo{1, 224, 64, 64},
    ConvLayerInfo{2, 112, 64, 128},  ConvLayerInfo{3, 112, 128, 128},
    ConvLayerInfo{4, 56, 128, 256},  ConvLayerInfo{5, 56, 256, 256},
    ConvLayerInfo{6, 56, 256, 256},  ConvLayerInfo{7, 28, 256, 512},
    ConvLayerInfo{8, 28, 512, 512},  ConvLayerInfo{9, 28, 512, 512},
    ConvLayerInfo{10, 14, 512, 512}, ConvLayerInfo{11, 14, 512, 512},
    ConvLayerInfo{12, 14, 512, 512}};

inline constexpr size_t kConvCount = kConvLayerInfos.size();

// order, input_count, output_count
inline constexpr std::array<DenseLayerInfo, 3> kDenseLayerInfos{
    DenseLayerInfo{0, 25088, 4096}, DenseLayerInfo{1, 4096, 4096},
    DenseLayerInfo{2, 4096, 1000}};
inline constexpr size_t kDenseCount = kDenseLayerInfos.size();

// order, D, C
inline constexpr std::array<ImageInfo, 38> kImageInfos{
    ImageInfo{0, 224, 3},   ImageInfo{1, 224, 64},  ImageInfo{2, 224, 64},
    ImageInfo{3, 224, 64},  ImageInfo{4, 224, 64},  ImageInfo{5, 112, 64},
    ImageInfo{6, 112, 128}, ImageInfo{7, 112, 128}, ImageInfo{8, 112, 128},
    ImageInfo{9, 112, 128}, ImageInfo{10, 56, 128}, ImageInfo{11, 56, 256},
    ImageInfo{12, 56, 256}, ImageInfo{13, 56, 256}, ImageInfo{14, 56, 256},
    ImageInfo{15, 56, 256}, ImageInfo{16, 56, 256}, ImageInfo{17, 28, 256},
    ImageInfo{18, 28, 512}, ImageInfo{19, 28, 512}, ImageInfo{20, 28, 512},
    ImageInfo{21, 28, 512}, ImageInfo{22, 28, 512}, ImageInfo{23, 28, 512},
    ImageInfo{24, 14, 512}, ImageInfo{25, 14, 512}, ImageInfo{26, 14, 512},
    ImageInfo{27, 14, 512}, ImageInfo{28, 14, 512}, ImageInfo{29, 14, 512},
    ImageInfo{30, 14, 512}, ImageInfo{31, 7, 512},  ImageInfo{32, 1, 25088},
    ImageInfo{33, 1, 4096}, ImageInfo{34, 1, 4096}, ImageInfo{35, 1, 4096},
    ImageInfo{36, 1, 4096}, ImageInfo{37, 1, 1000}};

inline constexpr size_t kImageCount = kImageInfos.size();

enum LayerType { kConv, kRelu, kPooling, kFlatten, kDense };

inline constexpr std::array<std::pair<LayerType, size_t>, 37> kLayerTypeOrders{
    {{kConv, 0},    {kRelu, 0},    {kConv, 1},  {kRelu, 1},    {kPooling, 0},
     {kConv, 2},    {kRelu, 2},    {kConv, 3},  {kRelu, 3},    {kPooling, 1},
     {kConv, 4},    {kRelu, 4},    {kConv, 5},  {kRelu, 5},    {kConv, 6},
     {kRelu, 6},    {kPooling, 2}, {kConv, 7},  {kRelu, 7},    {kConv, 8},
     {kRelu, 8},    {kConv, 9},    {kRelu, 9},  {kPooling, 3}, {kConv, 10},
     {kRelu, 10},   {kConv, 11},   {kRelu, 11}, {kConv, 12},   {kRelu, 12},
     {kPooling, 4}, {kFlatten, 0}, {kDense, 0}, {kRelu, 13},   {kDense, 1},
     {kRelu, 14},   {kDense, 2}}};

inline constexpr std::array<size_t, 13> kConvLayers{0,  2,  5,  7,  10, 12, 14,
                                                    17, 19, 21, 24, 26, 28};

inline constexpr std::array<size_t, 3> kDenseLayers{32, 34, 36};

inline constexpr std::array<size_t, 15> kReluLayers{
    1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29, 33, 35};

inline constexpr std::array<size_t, 5> kPoolingLayers{4, 9, 16, 23, 30};

//inline constexpr D = 8;
//inline constexpr N = 24;
}  // namespace clink::vgg16