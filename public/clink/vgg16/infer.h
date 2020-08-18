#pragma once

#include "./para_com.h"
#include "./para_fr.h"
#include "./para_pub.h"
#include "circuit/fixed_point/fixed_point.h"
#include "circuit/vgg16/vgg16.h"

namespace clink::vgg16 {

// input type: <D,N>
// output type: <D, 2N>
inline void InferConv(Para::ConvLayer const& layer, Image const& input_image,
                      Image& output_image) {
  Tick tick(__FN__, std::to_string(layer.order));
  // input_image.dump<8,24>();
  // layer.dump();
  namespace fp = circuit::fp;
  size_t const C = layer.C();
  size_t const D = layer.D;
  size_t const K = layer.K();
  auto DD = D * D;
  auto CDD = C * DD;
  auto KCDD = K * CDD;

  assert(input_image.D() == D);
  assert(input_image.C() == C);
  assert(output_image.D() == D);
  assert(output_image.C() == K);

  auto get_image = [&input_image](size_t h, size_t i, size_t j) -> Fr const& {
    auto d = input_image.D();
    if (i == 0 || j == 0 || i == (d + 1) || j == (d + 1)) return FrZero();
    return input_image.pixels[h][i - 1][j - 1];
  };

  boost::multi_array<Fr, 2> b(boost::extents[CDD][9]);
  auto pf1 = [D, DD, &b, &get_image](int64_t i) {
    for (size_t j = 0; j < 9; ++j) {
      size_t m = j / 3;
      size_t n = j % 3;
      size_t r = i / DD;
      size_t p = i % DD;
      size_t q = p / D;
      size_t o = p % D;
      b[i][j] = get_image(r, q + m, o + n);
    }
  };
  parallel::For(CDD, pf1);

  boost::multi_array<Fr, 2> c(boost::extents[KCDD][9]);
  auto pf2 = [CDD, &c, &b](int64_t i) {
    for (size_t j = 0; j < 9; ++j) {
      c[i][j] = b[i % CDD][j];
    }
  };
  parallel::For(KCDD, pf2);

  boost::multi_array<Fr, 2> p(boost::extents[KCDD][9]);
  auto pf3 = [DD, C, &layer, &p](int64_t i) {
    size_t coef_offset = i / DD;
    size_t coef_k = coef_offset / C;
    size_t coef_c = coef_offset % C;
    for (size_t j = 0; j < 9; ++j) {
      p[i][j] = layer.coefs[coef_k][coef_c][j / 3][j % 3];
    }
  };
  parallel::For(KCDD, pf3);

  std::vector<Fr> x(KCDD);
  auto pf4 = [&x, &c, &p](int64_t i) {
    x[i] = std::inner_product(c[i].begin(), c[i].end(), p[i].begin(), FrZero());
  };
  parallel::For(KCDD, pf4);

  auto& output = output_image.pixels;
  auto pf5 = [C, D, DD, CDD, &x, &output, &layer](int64_t i) {
    auto bias = layer.bias[i] * fp::RationalConst<8, 24>().kFrN;
    for (size_t j = 0; j < D; ++j) {
      for (size_t k = 0; k < D; ++k) {
        for (size_t l = 0; l < C; ++l) {
          output[i][j][k] += x[i * CDD + l * DD + j * D + k];
        }
        output[i][j][k] += bias;
      }
    }
  };
  parallel::For(K, pf5);

#ifdef _DEBUG
  std::vector<std::vector<std::vector<Fr>>> debug_output(K);
  for (size_t i = 0; i < K; ++i) {
    debug_output[i].resize(D);
    for (size_t j = 0; j < D; ++j) {
      debug_output[i][j].resize(D);
      for (size_t k = 0; k < D; ++k) {
        debug_output[i][j][k] = output[i][j][k];
      }
    }
  }
#endif
  std::cout << Tick::GetIndentString()
            << fp::RationalToDouble<8, 48>(output_image.data.front()) << " "
            << fp::RationalToDouble<8, 48>(output_image.data.back()) << "\n";

  return;
}

// input type: <D,2N>
// output type: <D,N>
inline void InferRelu(Image const& input_image, Image& output_image) {
  Tick tick(__FN__);
  namespace fp = circuit::fp;
  auto const& input_data = input_image.pixels;
  auto& output_data = output_image.pixels;

  auto pf = [&input_data, &output_data](int64_t i) {
    for (size_t j = 0; j < input_data[0].size(); ++j) {
      for (size_t k = 0; k < input_data[0][0].size(); ++k) {
        auto& in_data = input_data[i][j][k];
        auto& out_data = output_data[i][j][k];
        out_data = fp::ReducePrecision<8, 24 * 2, 24>(in_data);
        out_data = out_data.isNegative() ? 0 : out_data;

        if (DEBUG_CHECK) {
          libsnark::protoboard<Fr> pb;
          fp::Relu2Gadget<8, 24 * 2, 24> gadget(pb, "Relu2Gadget");
          gadget.Assign(in_data);
          CHECK(pb.is_satisfied(), "");
          Fr gadget_data = pb.val(gadget.ret());
          CHECK(gadget_data == out_data, "");
        }
      }
    }
  };
  parallel::For(input_data.size(), pf);

#ifdef _DEBUG
  std::vector<std::vector<std::vector<Fr>>> debug_data(output_data.size());
  for (size_t i = 0; i < output_data.size(); ++i) {
    debug_data[i].resize(output_data[0].size());
    for (size_t j = 0; j < output_data[0].size(); ++j) {
      debug_data[i][j].resize(output_data[0][0].size());
      for (size_t k = 0; k < output_data[0][0].size(); ++k) {
        debug_data[i][j][k] = output_data[i][j][k];
      }
    }
  }
#endif
  std::cout << Tick::GetIndentString()
            << fp::RationalToDouble<8, 24>(output_image.data.front()) << " "
            << fp::RationalToDouble<8, 24>(output_image.data.back()) << "\n";
  return;
}

// input type: <D,N>
// output type: <D,N>
inline void InferMaxPooling(Image const& input_image, Image& output_image) {
  Tick tick(__FN__);
  namespace fp = circuit::fp;
  auto const& input_data = input_image.pixels;
  auto& output_data = output_image.pixels;
  auto const& FrDN = fp::RationalConst<8, 24>().kFrDN;

  auto pf = [&input_data, &output_data, &FrDN](int64_t i) {
    for (size_t j = 0; j < output_data[0].size(); ++j) {
      for (size_t k = 0; k < output_data[0][0].size(); ++k) {
        std::array<Fr, 4> rect_fr;
        rect_fr[0] = input_data[i][j * 2][k * 2] + FrDN;
        rect_fr[1] = input_data[i][j * 2][k * 2 + 1] + FrDN;
        rect_fr[2] = input_data[i][j * 2 + 1][k * 2] + FrDN;
        rect_fr[3] = input_data[i][j * 2 + 1][k * 2 + 1] + FrDN;
        std::array<mpz_class, 4> rect_mpz;
        rect_mpz[0] = rect_fr[0].getMpz();
        rect_mpz[1] = rect_fr[1].getMpz();
        rect_mpz[2] = rect_fr[2].getMpz();
        rect_mpz[3] = rect_fr[3].getMpz();
        mpz_class max_value =
            *std::max_element(rect_mpz.begin(), rect_mpz.end());
        output_data[i][j][k].setMpz(max_value);
        output_data[i][j][k] = output_data[i][j][k] - FrDN;
      }
    }
  };
  parallel::For(output_data.size(), pf);

#ifdef _DEBUG
  std::vector<std::vector<std::vector<Fr>>> debug_data(output_data.size());
  for (size_t i = 0; i < output_data.size(); ++i) {
    debug_data[i].resize(output_data[0].size());
    for (size_t j = 0; j < output_data[0].size(); ++j) {
      debug_data[i][j].resize(output_data[0][0].size());
      for (size_t k = 0; k < output_data[0][0].size(); ++k) {
        debug_data[i][j][k] = output_data[i][j][k];
      }
    }
  }
#endif
  std::cout << Tick::GetIndentString()
            << fp::RationalToDouble<8, 24>(output_image.data.front()) << " "
            << fp::RationalToDouble<8, 24>(output_image.data.back()) << "\n";
  return;
}

// permute_dimension(1,2,0),that is 0,1,2->1,2,0
// [I,J,K]->[J,K,I]
inline void InferFlatten(Image const& input_image, Image& output_image) {
  namespace fp = circuit::fp;
  auto const& input_data = input_image.pixels;
  auto& output_data = output_image.pixels;
  auto I = input_data.size();
  auto J = input_data[0].size();
  auto K = input_data[0][0].size();
  for (size_t i = 0; i < I; ++i) {
    for (size_t j = 0; j < J; ++j) {
      for (size_t k = 0; k < K; ++k) {
        // o[j,k,i]=i[i,j,k]
        size_t index = j * K * I + k * I + i;
        output_data[index][0][0] = input_data[i][j][k];
      }
    }
  }
  std::cout << Tick::GetIndentString()
            << fp::RationalToDouble<8, 24>(output_image.data.front()) << " "
            << fp::RationalToDouble<8, 24>(output_image.data.back()) << "\n";
}

// input type: <D,N>
// output type: <D,2N>
inline void InferDense(Para::DenseLayer const& layer, Image const& input_image,
                       Image& output_image) {
  Tick tick(__FN__, std::to_string(layer.order));
  namespace fp = circuit::fp;
  auto const& input_data = input_image.pixels;
  auto& output_data = output_image.pixels;
  assert(input_data[0].size() == 1 && input_data[0][0].size() == 1);
  assert(output_data[0].size() == 1 && output_data[0][0].size() == 1);
  std::vector<Fr> input(input_data.size() + 1);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input[i] = input_data[i][0][0];
  }
  input.back() = fp::RationalConst<8, 24>().kFrN;

  auto pf = [&output_data, &input, &layer](int64_t i) {
    output_data[i][0][0] = std::inner_product(
        input.begin(), input.end(), layer.weight[i].begin(), FrZero());
  };
  parallel::For(output_data.size(), pf);
  std::cout << Tick::GetIndentString()
            << fp::RationalToDouble<8, 48>(output_image.data.front()) << " "
            << fp::RationalToDouble<8, 48>(output_image.data.back()) << "\n";
}

inline void Infer(Para const& para, dbl::Image const& dbl_image,
                  std::array<std::unique_ptr<Image>, kImageCount>& images) {
  Tick tick(__FN__);
  namespace fp = circuit::fp;
  images[0].reset(new Image(dbl_image));
  for (size_t i = 1; i < images.size(); ++i) {
    images[i].reset(new Image(kImageInfos[i]));
  }

  InferConv(para.conv_layer(0), *images[0], *images[1]);

  InferRelu(*images[1], *images[2]);

  InferConv(para.conv_layer(1), *images[2], *images[3]);

  InferRelu(*images[3], *images[4]);

  InferMaxPooling(*images[4], *images[5]);

  InferConv(para.conv_layer(2), *images[5], *images[6]);

  InferRelu(*images[6], *images[7]);

  InferConv(para.conv_layer(3), *images[7], *images[8]);

  InferRelu(*images[8], *images[9]);

  InferMaxPooling(*images[9], *images[10]);

  InferConv(para.conv_layer(4), *images[10], *images[11]);

  InferRelu(*images[11], *images[12]);

  InferConv(para.conv_layer(5), *images[12], *images[13]);

  InferRelu(*images[13], *images[14]);

  InferConv(para.conv_layer(6), *images[14], *images[15]);

  InferRelu(*images[15], *images[16]);

  InferMaxPooling(*images[16], *images[17]);

  InferConv(para.conv_layer(7), *images[17], *images[18]);

  InferRelu(*images[18], *images[19]);

  InferConv(para.conv_layer(8), *images[19], *images[20]);

  InferRelu(*images[20], *images[21]);

  InferConv(para.conv_layer(9), *images[21], *images[22]);

  InferRelu(*images[22], *images[23]);

  InferMaxPooling(*images[23], *images[24]);

  InferConv(para.conv_layer(10), *images[24], *images[25]);

  InferRelu(*images[25], *images[26]);

  InferConv(para.conv_layer(11), *images[26], *images[27]);

  InferRelu(*images[27], *images[28]);

  InferConv(para.conv_layer(12), *images[28], *images[29]);

  InferRelu(*images[29], *images[30]);

  InferMaxPooling(*images[30], *images[31]);

  InferFlatten(*images[31], *images[32]);

  InferDense(para.dense_layer(0), *images[32], *images[33]);

  InferRelu(*images[33], *images[34]);

  InferDense(para.dense_layer(1), *images[34], *images[35]);

  InferRelu(*images[35], *images[36]);

  InferDense(para.dense_layer(2), *images[36], *images[37]);

  // images[37]->dump<8, 48>();
  std::cout << Tick::GetIndentString()
            << fp::RationalToDouble<8, 48>(images[37]->data.front()) << " "
            << fp::RationalToDouble<8, 48>(images[37]->data.back()) << "\n";
}

};  // namespace clink::vgg16