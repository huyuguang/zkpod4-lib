#pragma once

#include "para_pub.h"
#include "public.h"

namespace clink {
namespace vgg16 {
namespace dbl {
class Para {
 public:
  Para(std::string const& path) : path_(path) {
    Tick tick(__FN__);
    CHECK(LoadConvLayers(), path);
    CHECK(LoadDenseLayers(), path);
  }

  struct ConvLayer {
    ConvLayer(ConvLayerInfo const& info) : order(info.order), D(info.D) {
      coefs.resize(info.K);
      for (auto& i : coefs) {
        i.resize(info.C);
      }
      bias.resize(info.K);
    }
    size_t const order;
    size_t const D;
    // size=K*C*3*3
    std::vector<std::vector<std::array<std::array<double, 3>, 3>>> coefs;
    std::vector<double> bias;  // size=K
    size_t K() const { return bias.size(); }
    size_t C() const { return coefs[0].size(); }
  };

  struct DenseLayer {
    DenseLayer(DenseLayerInfo const& info) : order(info.order) {
      weight.resize(info.type_count);
      for (auto& i : weight) {
        i.resize(info.input_count + 1);
      }
    }
    size_t const order;
    std::vector<std::vector<double>> weight;
  };

  ConvLayer const& conv_layer(size_t order) const {
    return *conv_layers_[order];
  }

  DenseLayer const& dense_layer(size_t order) const {
    return *dense_layers_[order];
  }

 private:
  bool LoadConvLayers() {
    Tick tick(__FN__);
    for (size_t i = 0; i < conv_layers_.size(); ++i) {
      conv_layers_[i].reset(new ConvLayer(kConvLayerInfos[i]));
      if (!LoadConvLayer(*conv_layers_[i])) {
        return false;
      }
    }
    return true;
  }

  bool LoadConvLayer(ConvLayer& layer) {
    std::string order_str = std::to_string(layer.order + 1);
    std::string conv_name = path_ + "/features_conv_" + order_str + "/" +
                            "conv" + order_str + ".txt";
    std::ifstream conv_file(conv_name);
    std::string bias_name =
        path_ + "/features_conv_" + order_str + "/" + "bias.txt";
    std::ifstream bias_file(bias_name);

    std::string conv_line;
    std::string bias_line;
    for (size_t k = 0; k < layer.bias.size(); ++k) {
      auto& bias = layer.bias[k];
      auto& coefs = layer.coefs[k];
      if (!std::getline(conv_file, conv_line)) {
        assert(false);
        return false;
      }
      if (conv_line != "conv" + order_str + "_ " + std::to_string(k)) {
        assert(false);
        return false;
      }

      for (size_t c = 0; c < coefs.size(); ++c) {
        auto& coef = coefs[c];

        for (size_t i = 0; i < 3; ++i) {
          if (!std::getline(conv_file, conv_line)) {
            assert(false);
            return false;
          }
          for (auto& s : conv_line) {
            if (s == '\t' || s == '[' || s == ']' || s == '\r' || s == '\n')
              s = ' ';
          }
          std::istringstream iss(conv_line);
          if (!(iss >> coef[i][0] >> coef[i][1] >> coef[i][2])) {
            assert(false);
            return false;
          }
        }
      }

      if (!std::getline(bias_file, bias_line)) {
        assert(false);
        return false;
      }
      std::istringstream iss(bias_line);
      if (!(iss >> bias)) {
        assert(false);
        return false;
      }
    }

    return true;
  }

  bool LoadDenseLayers() {
    Tick tick(__FN__);
    for (size_t i = 0; i < dense_layers_.size(); ++i) {
      dense_layers_[i].reset(new DenseLayer(kDenseLayerInfos[i]));
      if (!LoadDenseLayer(*dense_layers_[i])) {
        return false;
      }
    }
    return true;
  }

  bool LoadDenseLayer(DenseLayer& layer) {
    std::string order_str = std::to_string(layer.order + 1);

    std::string bias_name =
        path_ + "/features_dense_" + order_str + "/" + "bias.txt";
    std::ifstream bias_file(bias_name);

    std::string weight_name =
        path_ + "/features_dense_" + order_str + "/" + "weights.txt";
    std::ifstream weight_file(weight_name);

    for (size_t c = 0; c < layer.weight.size(); ++c) {
      std::string line;
      if (!std::getline(bias_file, line)) {
        assert(false);
        return false;
      }

      std::istringstream iss(line);
      iss >> layer.weight[c].back();
    }

    for (size_t c = 0; c < layer.weight[0].size() - 1; ++c) {
      std::string line;
      if (!std::getline(weight_file, line)) {
        assert(false);
        return false;
      }

      for (auto& s : line) {
        if (s == '\t' || s == ',' || s == ';' || s == '\r' || s == '\n')
          s = ' ';
      }

      std::istringstream iss(line);
      for (size_t d = 0; d < layer.weight.size(); ++d) {
        iss >> layer.weight[d][c];
      }
    }
    return true;
  }

 private:
  std::string const path_;
  std::array<std::unique_ptr<ConvLayer>, kConvCount> conv_layers_;
  std::array<std::unique_ptr<DenseLayer>, kDenseCount> dense_layers_;
};

struct Image {
  size_t const order;
  std::vector<double> data;
  boost::multi_array_ref<double, 3> pixels;

  Image(ImageInfo const& info)
      : order(info.order),
        data(info.size()),
        pixels(data.data(), boost::extents[info.C][info.D][info.D]) {}

  size_t size() const { return data.size(); }
  size_t C() const { return pixels.size(); }
  size_t D() const { return pixels[0].size(); }
  void dump() const {
    for (auto i : data) {
      std::cout << i << ";";
    }
    std::cout << "\n\n";
  }
};

inline bool LoadTestImage(std::string const& path, Image& image) {
  assert(image.C() == 3);
  std::string line;
  for (size_t c = 0; c < image.C(); ++c) {
    auto name = path + "/test_image_" + std::to_string(c) + ".txt";
    std::ifstream file(name);
    for (size_t i = 0; i < image.D(); ++i) {
      if (!std::getline(file, line)) {
        std::cerr << "read file " << path << " failed\n";
        assert(false);
        return false;
      }

      for (auto& s : line) {
        if (s == ',') s = ' ';
      }

      std::istringstream iss(line);
      for (size_t j = 0; j < image.D(); ++j) {
        if (!(iss >> image.pixels[c][i][j])) {
          std::cerr << "read file " << path << " failed, data invalid\n";
          assert(false);
          return false;
        }
      }
    }
  }
  return true;
}

inline void InferConv(Para::ConvLayer const& layer, Image const& input_image,
                      Image& output_image) {
  Tick tick(__FN__, std::to_string(layer.order));
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

  auto get_image = [&input_image](size_t h, size_t i, size_t j) -> double {
    auto d = input_image.D();
    if (i == 0 || j == 0 || i == (d + 1) || j == (d + 1)) return 0;
    return input_image.pixels[h][i - 1][j - 1];
  };

  boost::multi_array<double, 2> b(boost::extents[CDD][9]);
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

  boost::multi_array<double, 2> c(boost::extents[KCDD][9]);
  auto pf2 = [CDD, &c, &b](int64_t i) {
    for (size_t j = 0; j < 9; ++j) {
      c[i][j] = b[i % CDD][j];
    }
  };
  parallel::For(KCDD, pf2);

  boost::multi_array<double, 2> p(boost::extents[KCDD][9]);
  auto pf3 = [DD, C, &layer, &p](int64_t i) {
    size_t coef_offset = i / DD;
    size_t coef_k = coef_offset / C;
    size_t coef_c = coef_offset % C;
    for (size_t j = 0; j < 9; ++j) {
      p[i][j] = layer.coefs[coef_k][coef_c][j / 3][j % 3];
    }
  };
  parallel::For(KCDD, pf3);
  
  std::vector<double> x(KCDD);
  auto pf4 = [&x, &c, &p](int64_t i) {
    x[i] = std::inner_product(c[i].begin(), c[i].end(), p[i].begin(), 0.0);
  };
  parallel::For(KCDD, pf4);

  auto& output = output_image.pixels;
  auto pf5 = [C, D, DD, CDD, &x, &output, &layer](int64_t i) {
    for (size_t j = 0; j < D; ++j) {
      for (size_t k = 0; k < D; ++k) {
        for (size_t l = 0; l < C; ++l) {
          output[i][j][k] += x[i * CDD + l * DD + j * D + k];
        }
        output[i][j][k] += layer.bias[i];
      }
    }
  };
  parallel::For(K, pf5);

#ifdef _DEBUG
  std::vector<std::vector<std::vector<double>>> debug_output(K);
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
  std::cout << output_image.data.front() << " " << output_image.data.back()
            << "\n";
  return;
}

inline void InferRelu(Image const& input_image, Image& output_image) {
  Tick tick(__FN__);
  auto const& input_data = input_image.pixels;
  auto& output_data = output_image.pixels;

  auto pf = [&input_data, &output_data](int64_t i) {
    for (size_t j = 0; j < input_data[0].size(); ++j) {
      for (size_t k = 0; k < input_data[0][0].size(); ++k) {
        output_data[i][j][k] =
            input_data[i][j][k] < 0 ? 0 : input_data[i][j][k];
      }
    }
  };
  parallel::For(input_data.size(), pf);

#ifdef _DEBUG
  std::vector<std::vector<std::vector<double>>> debug_data(output_data.size());
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
  std::cout << output_image.data.front() << " " << output_image.data.back()
            << "\n";
  return;
}

inline void InferMaxPooling(Image const& input_image, Image& output_image) {
  Tick tick(__FN__);
  auto const& input_data = input_image.pixels;
  auto& output_data = output_image.pixels;

  auto pf = [&input_data,&output_data](int64_t i) {
    for (size_t j = 0; j < output_data[0].size(); ++j) {
      for (size_t k = 0; k < output_data[0][0].size(); ++k) {
        std::array<double, 4> rect_data;
        rect_data[0] = input_data[i][j * 2][k * 2];
        rect_data[1] = input_data[i][j * 2][k * 2 + 1];
        rect_data[2] = input_data[i][j * 2 + 1][k * 2];
        rect_data[3] = input_data[i][j * 2 + 1][k * 2 + 1];
        output_data[i][j][k] =
            *std::max_element(rect_data.begin(), rect_data.end());
      }
    }
  };
  parallel::For(output_data.size(), pf);

#ifdef _DEBUG
  std::vector<std::vector<std::vector<double>>> debug_data(output_data.size());
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

  if (output_image.C() == 512 && output_image.D() == 7) {
    for (size_t i = 0; i < output_image.C(); ++i) {
      std::cout << i << "\n";
      for (size_t j = 0; j < output_image.D(); ++j) {
        for (size_t k = 0; k < output_image.D(); ++k) {
          std::cout << output_image.pixels[i][j][k] << ",";
        }
        std::cout << "\n";
      }
      std::cout << "\n\n\n";
    }
  }

  //std::cout << output_image.data.front() << " " << output_image.data.back()
  //          << "\n";
  return;
}

// permute_dimension(1,2,0),that is 0,1,2->1,2,0
// [I,J,K]->[J,K,I]
inline void InferFlatten(Image const& input_image, Image& output_image) {
  Tick tick(__FN__);
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
  output_image.dump();
  std::cout << "\n";
  std::cout << output_image.data.front() << " " << output_image.data.back()
            << "\n";
}

inline void InferDense(Para::DenseLayer const& layer, Image const& input_image,
                       Image& output_image) {
  Tick tick(__FN__, std::to_string(layer.order));
  auto const& input_data = input_image.pixels;
  auto& output_data = output_image.pixels;
  assert(input_data[0].size() == 1 && input_data[0][0].size() == 1);
  assert(output_data[0].size() == 1 && output_data[0][0].size() == 1);
  std::vector<double> input(input_data.size() + 1);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input[i] = input_data[i][0][0];
  }
  input.back() = 1.0;

  auto pf = [&output_data, &input, &layer](int64_t i) {
    output_data[i][0][0] = std::inner_product(input.begin(), input.end(),
                                              layer.weight[i].begin(), 0.0);
  };
  parallel::For(output_data.size(), pf);
  std::cout << output_image.data.front() << " " << output_image.data.back()
            << "\n";
}

inline void Test(std::string const& para_path,
                 std::string const& test_image_path) {
  Tick tick(__FN__);
  Para para(para_path);

  std::array<std::unique_ptr<Image>, kImageCount> images;
  for (size_t i = 0; i < images.size(); ++i) {
    images[i].reset(new Image(kImageInfos[i]));
  }

  LoadTestImage(test_image_path, *images[0]);

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

  images[37]->dump();

  uint32_t max_value = 0;
  for (auto const& image : images) {
    auto const& pixels = image->pixels;
    for (size_t i = 0; i < pixels.size(); ++i) {
      for (size_t j = 0; j < pixels[0].size(); ++j) {
        for (size_t k = 0; k < pixels[0][0].size(); ++k) {
          if (uint32_t(std::abs(pixels[i][j][k])) > max_value) {
            max_value = uint32_t(std::abs(pixels[i][j][k]));
          }
        }
      }
    }
  }
  std::cout << "max_value: " << max_value;  // 176
}
}  // namespace dbl
}  // namespace vgg16
}  // namespace clink