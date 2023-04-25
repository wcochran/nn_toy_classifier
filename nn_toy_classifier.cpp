#include <vector>
#include <memory>
#include <cassert>
#include <Eigen/Dense>
#include <random>
#include <functional>

class Module {
public:
    enum class ModuleType {
        Linear,
        Sigmoid,
        ReLU,
        SoftMax,
        Sequential,
        MSELoss
    };
    const ModuleType _type;
    const size_t _numInputs, _numOutputs;
    Module *_prev;  // weak ptr to previous layer (nullptr => input layer)
    Module(ModuleType t, size_t in, size_t out) : _type{t}, _numInputs{in}, _numOutputs{out}, _prev{nullptr} {}
    virtual ~Module() {};
    size_t numInputs() const {return _numInputs;}
    size_t numOutputs() const {return _numOutputs;}
    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& input) = 0;
    virtual const Eigen::VectorXf& output() const = 0;

    void setPrevious(Module* prev) {_prev = prev;}
    virtual void zeroGrad() {};
    virtual void backward(const Eigen::VectorXf& grad) = 0;
    virtual void update(float learningRate) {};
};

class LinearModule : public Module { // fully connected
    Eigen::VectorXf _x;   // cached input (+ homogenous 1 added)
    Eigen::MatrixXf _W;   // weights + bias
    Eigen::VectorXf _z;   // cached output
    Eigen::MatrixXf _dW;  // weight + bias gradients
    size_t _numGradients; // number of accumulate gradients in _dW
public:
    LinearModule(size_t in, size_t out)
        : Module(ModuleType::Linear, in, out), _x(in), _W(out,in+1), _z(out),
          _dW(out,in+1), _numGradients{0} {}
    
    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& x) {
        assert(x.rows() == numInputs());
        _x = x.homogeneous();
        _z.noalias() = _W * _x;
        return _z;
    }
    virtual const Eigen::VectorXf& output() const {return _z;}

    //
    // The last column are the biases
    //
    Eigen::MatrixXf& weights() { return _W; }
    
    virtual void zeroGrad() {
        _numGradients = 0;
        _dW.setZero();
    }

    //
    // We are given the input accumulated gradients {dL/dz_i} for each
    // of our outputs z_i which have propagated backwards to the
    // output of this layer.
    // N = number of outputs
    // M = number of inputs
    // i = index for output layer z_i  (i = 0 .. N-1)
    // j = index for input later x_j   (j = 0 .. M-1)
    // W is N x M  [ W_ij ]
    // For updating our local parameter gradients we need:
    //       dL/dW_{ij} = dL/dz_i * dz_i/dW_{ij} = dL/dz_i * x_j
    //                     _           _   _                    _
    //       dL/dW      = |   dL/dz_0   | | x_0  x_1 ... x_{M-1} |
    //                    |   dL/dz_1   |  -                    -
    //                    |      .      |
    //                    | dL/dz_{N-1} |  (note: no dependence on W)
    //                     -           _
    // For continued back propogation to the prevous layer we need:
    //       dL/dx_j = Sum_i dL/dz_i * dz_i/dx_j
    //               = Sum_i dL/dz_i * W_{ij}
    //       dL/dx   = [dL/dz_0 dL/dz_1 ... dL/dz_{N-1}] * W
    //
    virtual void backward(const Eigen::VectorXf& dz) {
        assert(dz.size() == numOutputs());
        const Eigen::MatrixXf dW = dz * _x.transpose();
        _dW += dW;
        _numGradients++;
        if (_prev != nullptr) {
            const Eigen::VectorXf dx = dz.transpose() * _W;
            _prev->backward(dx);
        }
    }
    
    virtual void update(float learningRate) {
        assert(_numGradients > 0);
        _W -= learningRate / _numGradients * _dW; // step in negative dir of gradient
    }
};

class SigmoidModule : public Module {
    Eigen::VectorXf _z; // cached output
    SigmoidModule(size_t inOut) : Module(ModuleType::Sigmoid, inOut, inOut), _z(inOut) {}
public:
    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& x) {
        _z.noalias() = x.unaryExpr([](float elem) -> float {
            const float ex = std::exp(elem);
            return ex/(ex + 1);
        });
        return _z;
    }
    virtual const Eigen::VectorXf& output() const {return _z;}
    virtual void backward(const Eigen::VectorXf& grad) {
        if (_prev != nullptr) {
            assert(grad.size() == numOutputs());
            const Eigen::VectorXf one = Eigen::VectorXf::Constant(numOutputs(),1.0f);
            const Eigen::VectorXf dzdx = _z.cwiseProduct(one - _z); // s' = s*(1 - s)
            const Eigen::VectorXf dLdz = grad.cwiseProduct(dzdx);
            _prev->backward(dLdz);
        }
    }
};

class ReLUModule : public Module {
    Eigen::VectorXf _a; // cached output
public:
    ReLUModule(size_t inOut) : Module(ModuleType::ReLU, inOut, inOut), _a(inOut) {}
    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& x) {
        _a.noalias() = x.unaryExpr([](float elem) -> float {
            return std::max(0.f,elem);
        });
        return _a;
    }
    virtual const Eigen::VectorXf& output() const {return _a;}

    virtual void backward(const Eigen::VectorXf& grad) {
        if (_prev != nullptr) {
            assert(grad.size() == numOutputs());
            const Eigen::VectorXf dzdx = _a.unaryExpr([](float a) -> float {
                return (a > 0) ? 1.0f : 0.0f;
            });
            const Eigen::VectorXf dLdz = grad.cwiseProduct(dzdx);
            _prev->backward(dLdz);
        }
    }
};

class SoftMaxModule : public Module {
    Eigen::Vector<float, Eigen::Dynamic> _a;
public:
    SoftMaxModule(size_t inOut) : Module(ModuleType::SoftMax, inOut, inOut), _a(inOut) {}
    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& x) {
        assert(x.rows() == numInputs());
        const Eigen::VectorXf e2x = x.unaryExpr([](float elem) -> float {
            return std::exp(elem);
        });
        const float sum = e2x.sum();
        _a.noalias() = e2x.unaryExpr([sum](float elem) -> float {
            return elem/sum;
        });
        return _a;
    }
    virtual const Eigen::VectorXf& output() const {return _a;}
    virtual void backward(const Eigen::VectorXf& grad) {
        assert(false); // unimplemented
    }
};

class SequentialModule : public Module {
    std::vector<std::shared_ptr<Module>> _modules;
public:
    SequentialModule(std::vector<std::shared_ptr<Module>>& modules)
        : Module(ModuleType::Sequential, modules.front()->numInputs(), modules.back()->numOutputs()),
          _modules(modules) {
        for (size_t i = 1; i < _modules.size(); i++)
            _modules[i]->setPrevious(_modules[i-1].get());
    }

    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& x) {
        (*_modules[0])(x);
        for (size_t i = 1; i < _modules.size(); i++) {
            (*_modules[i])(_modules[i-1]->output());
        }
        return _modules.back()->output();
    }
    virtual const Eigen::VectorXf& output() const {
        return _modules.back()->output();
    }

    virtual void zeroGrad() {
        for (auto&& m : _modules)
            m->zeroGrad();
    }
    
    virtual void backward(const Eigen::VectorXf& grad) {
        assert(grad.size() == _modules.back()->numOutputs());
        _modules.back()->backward(grad);
    }
    
    virtual void update(float learningRate) {
        for (size_t i = 0; i < _modules.size(); i++)
            _modules[i]->update(learningRate);
    }
};

class MSELossModule : public Module {
    Eigen::VectorXf _x, _y; // cached input
    float _loss;            // cached output
    Eigen::VectorXf _vloss; // loss as vector
public:
    MSELossModule(size_t in) : Module(ModuleType::MSELoss, in, 1), _x(in), _loss{}, _vloss(1) {}
    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& x) {
        assert(false);  // not used for loss functions
        static Eigen::VectorXf nada = Eigen::VectorXf::Zero(1);  // not reached;
        return nada;
    }
    virtual const float operator()(const Eigen::VectorXf& x,
                                   const Eigen::VectorXf& y) {
        _x = x; // computed result
        _y = y; // target result
        _loss = 0.5 * (x - y).squaredNorm();
        _vloss(0) = _loss;
        return _loss;
    }
    virtual const Eigen::VectorXf& output() const { return _vloss; }
    virtual void backward(const Eigen::VectorXf& dz) {
        assert(false);
    }
    virtual void backward() {
        if (_prev == nullptr) return;
        const Eigen::VectorXf dz = _x - _y;
        _prev->backward(dz);
    }
};

void kaimingNormalInit(Eigen::MatrixXf& W) {
    const auto N = W.cols() - 1; // num inputs
    const auto M = W.rows();     // num outputs
    const float var = 2.0/(N + M);
    std::mt19937 gen(1234);
    std::normal_distribution<> dist{0.0f, var};
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            W(i,j) = dist(gen);
    W.col(N) = Eigen::VectorXf::Zero(M); // biases
}

#include <iostream>

void celcius_to_farenheit() {

    //
    // Create some input values and labels.
    //
    std::mt19937 gen(1234);
    constexpr float lo_c = 0, hi_c = 100;
    std::uniform_real_distribution<float> dist(lo_c, hi_c);
    
    constexpr size_t n = 10;
    Eigen::VectorXf F(n);
    Eigen::VectorXf C(n);
    for (size_t i = 0; i < n; i++) {
        const float c = dist(gen);
        const float f = 9.0f/5 * c + 32.0f;
        C(i) = c;
        F(i) = f;
    }

    //
    // Normalize input using mean and stdev of
    // uniform distribution on interval [a,b].
    //
    constexpr float c_mean = (lo_c + hi_c)/2;
    constexpr float sqrt12 = 3.4641016151f;
    constexpr float c_stdev = (hi_c - lo_c)/sqrt12;
    Eigen::VectorXf Cnorm(n);
    for (size_t i = 0; i < n; i++) {
        Cnorm(i) = (C(i) - c_mean)/c_stdev;
    }

    //
    // Create MLP with one linear layer.
    //
    std::vector<std::shared_ptr<Module>> modules;
    auto linearLayer = std::make_shared<LinearModule>(1, 1);
    modules.emplace_back(linearLayer);
    SequentialModule converter(modules);

    //
    // Create loss function
    //
    MSELossModule lossFunc(converter.numOutputs());
    lossFunc.setPrevious(&converter);

    //
    // Initialize linear module weights according to
    // Kaiming He algorithm (mimic torch.nn.init.kaiming_normal)
    // This is typically used when the linear layer is followed
    // by a ReLU function (which we don't have).
    //
    kaimingNormalInit(linearLayer->weights());

    //
    // Train.
    // Batch size = Input size.
    //
    constexpr size_t numEpochs = 70;
    constexpr float learningRate = 0.5;
    for (size_t epoch = 0; epoch < numEpochs; epoch++) {
        converter.zeroGrad();
        float totalLoss = 0;
        for (size_t i = 0; i < n; i++) {
            const Eigen::VectorXf x = Cnorm.row(i);
            const Eigen::VectorXf y = converter(x);
            const Eigen::VectorXf f = F.row(i);
            const float loss = lossFunc(y,f);
            totalLoss += loss;
            lossFunc.backward();
        }
        const float averageLoss = totalLoss / n;
        std::cout << "epoch:" << epoch << ": loss = " << averageLoss << "\n";
        converter.update(learningRate);
    }

    //
    // Trained celsius to farenheit converter.
    //
    auto celciusToFarenhright = [&](float celcius) -> float {
        const float cnorm = (celcius - c_mean)/c_stdev;
        Eigen::VectorXf cvec(1);
        cvec(0) = cnorm;
        const Eigen::VectorXf y = converter(cvec);
        return y(0);
    };

    //
    // Validation with known truth.
    //
    constexpr size_t nsamples = 20;
    const float dc = (hi_c - lo_c)/(nsamples - 1);
    for (size_t i = 0; i < nsamples; i++) {
        const float c = lo_c + i*dc;
        const float f = celciusToFarenhright(c);
        const float f_truth = 9.0/5 * c + 32;
        const float error = std::fabs(f - f_truth);
        std::cout << c << " " << f << " " << error << "\n";
    }

}

int main(int argc, char *argv[]) {
    celcius_to_farenheit();
    return 0;
}

